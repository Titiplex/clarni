#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <utility>

namespace ml
{
    struct Tensor::Buffer {
        std::vector<double> values;
        double* device_values=nullptr;
        int device=-1;
        ~Buffer(){if(device_values)gpu::resident::release(device_values,device);}
    };

    namespace
    {
        double sigmoid(double x)
        {
            if (x >= 0)return 1.0 / (1.0 + std::exp(-x));
            const double e = std::exp(x);
            return e / (1.0 + e);
        }

        std::size_t checked_size(int rows, int cols)
        {
            if (rows < 0 || cols < 0)throw std::invalid_argument("Tensor: negative shape");
            const auto r = static_cast<std::size_t>(rows), c = static_cast<std::size_t>(cols);
            if (c && r > std::vector<double>().max_size() / c)throw std::length_error("Tensor too large");
            return r * c;
        }
    }

    Tensor::Tensor(int rows, int cols, double value, Backend backend)
        : buffer_(std::make_shared<Buffer>()), rows_(rows), cols_(cols),
          backend_(backend == Backend::Auto ? compute_device().selected : backend)
    {
        const auto count=checked_size(rows,cols);
        if(backend_==Backend::CPU)buffer_->values.assign(count,value);
        else if(backend_==Backend::CUDA){
            if(compute_device().selected!=Backend::CUDA||!compute_device().cuda_usable)throw std::runtime_error("Initialize a usable CUDA device first");
            buffer_->device=compute_device().cuda_device;
            buffer_->device_values=gpu::resident::allocate(count,buffer_->device);
            gpu::resident::fill(buffer_->device_values,count,value,buffer_->device);
        }else throw std::invalid_argument("Unknown tensor backend");
    }
    const double* Tensor::read()const{if(backend_!=Backend::CPU)throw std::logic_error("GPU tensor cannot be dereferenced on CPU");return buffer_->values.data();}
    double* Tensor::write(){if(backend_!=Backend::CPU)throw std::logic_error("GPU tensor cannot be dereferenced on CPU");return buffer_->values.data();}
    void Tensor::compatible(const Tensor& other, bool shape) const
    {
        if(backend_!=other.backend_||(backend_==Backend::CUDA&&buffer_->device!=other.buffer_->device)||(shape&&(rows_!=other.rows_||cols_!=other.cols_)))
            throw std::invalid_argument("Tensor: incompatible shape/backend");
    }

    Tensor Tensor::device_operation(gpu::resident::Operation operation,int output_rows,int output_cols,const Tensor* b,const Tensor* c,int other,int auxiliary,double scale,double offset)const{
        Tensor result(output_rows,output_cols,0,backend_);
        gpu::resident::operation(operation,buffer_->device_values,b?b->buffer_->device_values:nullptr,c?c->buffer_->device_values:nullptr,
            result.buffer_->device_values,rows_,cols_,output_rows,output_cols,other,auxiliary,scale,offset,buffer_->device);
        return result;
    }

    Tensor Tensor::from_matrix(const math::Matrix& matrix, Backend backend)
    {
        Tensor result(matrix.get_x(), matrix.get_y(), 0, backend);
        std::vector<double> values(result.size());
        for(int i=0;i<result.rows_;++i)for(int j=0;j<result.cols_;++j)values[static_cast<std::size_t>(i)*result.cols_+j]=matrix(i,j);
        if(result.backend_==Backend::CPU)result.buffer_->values=std::move(values);
        else gpu::resident::upload(result.buffer_->device_values,values.data(),values.size(),result.buffer_->device);
        return result;
    }

    math::Matrix Tensor::to_matrix()const{
        math::Matrix result(rows_,cols_);std::vector<double> downloaded;
        const double* values=nullptr;
        if(backend_==Backend::CPU)values=read();
        else {downloaded.resize(size());gpu::resident::download(downloaded.data(),buffer_->device_values,size(),buffer_->device);values=downloaded.data();}
        for(int i=0;i<rows_;++i)for(int j=0;j<cols_;++j)result(i,j)=values[static_cast<std::size_t>(i)*cols_+j];
        return result;
    }

    Tensor Tensor::to(Backend backend) const
    {
        if (backend == Backend::Auto)backend = compute_device().selected;
        if (backend == backend_)return *this;
        return from_matrix(to_matrix(), backend);
    }

    Tensor Tensor::operator+(const Tensor& other) const
    {
        compatible(other);if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::Add,rows_,cols_,&other);
        Tensor out(rows_,cols_,0,backend_);
        for (std::size_t i = 0; i < size(); ++i)out.write()[i] = read()[i] + other.read()[i];
        return out;
    }

    Tensor Tensor::operator-(const Tensor& other) const
    {
        compatible(other);if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::Subtract,rows_,cols_,&other);
        Tensor out(rows_,cols_,0,backend_);
        for (std::size_t i = 0; i < size(); ++i)out.write()[i] = read()[i] - other.read()[i];
        return out;
    }

    Tensor Tensor::hadamard(const Tensor& other) const
    {
        compatible(other);if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::Multiply,rows_,cols_,&other);
        Tensor out(rows_,cols_,0,backend_);
        for (std::size_t i = 0; i < size(); ++i)out.write()[i] = read()[i] * other.read()[i];
        return out;
    }

    Tensor Tensor::divide(const Tensor& other) const
    {
        compatible(other);if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::Divide,rows_,cols_,&other);
        Tensor out(rows_,cols_,0,backend_);
        for (std::size_t i = 0; i < size(); ++i)out.write()[i] = read()[i] / other.read()[i];
        return out;
    }

    Tensor Tensor::affine(double scale, double offset) const
    {
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::Affine,rows_,cols_,nullptr,nullptr,0,0,scale,offset);
        Tensor out(rows_, cols_, 0, backend_);
        for (std::size_t i = 0; i < size(); ++i)out.write()[i] = scale * read()[i] + offset;
        return out;
    }

    Tensor Tensor::unary(Unary operation) const
    {
        if(operation<Unary::Exp||operation>Unary::Reciprocal)throw std::invalid_argument("Unknown unary operation");
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::Unary,rows_,cols_,nullptr,nullptr,0,static_cast<int>(operation));
        Tensor out(rows_, cols_, 0, backend_);
        for (std::size_t i = 0; i < size(); ++i)
        {
            const double x = read()[i];
            double y = 0;
            switch (operation)
            {
            case Unary::Exp: y = std::exp(x);
                break;
            case Unary::Log: y = std::log(x);
                break;
            case Unary::Square: y = x * x;
                break;
            case Unary::Sqrt: y = std::sqrt(x);
                break;
            case Unary::Reciprocal: y = 1 / x;
                break;
            }
            out.write()[i] = y;
        }
        return out;
    }

    Tensor Tensor::transpose() const
    {
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::Transpose,cols_,rows_);
        Tensor out(cols_, rows_, 0, backend_);
        for (int i = 0; i < rows_; ++i)for (int j = 0; j < cols_; ++j)out.write()[static_cast<std::size_t>(j) * rows_ +
            i] = read()[static_cast<std::size_t>(i) * cols_ + j];
        return out;
    }

    Tensor Tensor::matmul(const Tensor& other) const
    {
        compatible(other, false);
        if (cols_ != other.rows_)throw std::invalid_argument("Tensor matmul dimensions");
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::Matmul,rows_,other.cols_,&other);
        Tensor out(rows_, other.cols_, 0, backend_);
        for (int i = 0; i < rows_; ++i)
            for (int j = 0; j < other.cols_; ++j)
            {
                double sum = 0;
                for (int k = 0; k < cols_; ++k)sum += read()[static_cast<std::size_t>(i) * cols_ + k] * other.read()[
                    static_cast<std::size_t>(k) * other.cols_ + j];
                out.write()[static_cast<std::size_t>(i) * other.cols_ + j] = sum;
            }
        return out;
    }

    Tensor Tensor::sum_columns() const
    {
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::SumColumns,rows_,1);
        Tensor out(rows_, 1, 0, backend_);
        for (int i = 0; i < rows_; ++i)for (int j = 0; j < cols_; ++j)out.write()[i] += read()[static_cast<std::size_t>(
            i) * cols_ + j];
        return out;
    }

    Tensor Tensor::sum_rows() const
    {
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::SumRows,1,cols_);
        Tensor out(1, cols_, 0, backend_);
        for (int i = 0; i < rows_; ++i)for (int j = 0; j < cols_; ++j)out.write()[j] += read()[static_cast<std::size_t>(
            i) * cols_ + j];
        return out;
    }

    Tensor Tensor::broadcast_columns(int count) const
    {
        if (cols_ != 1 || count < 0)throw std::invalid_argument("Tensor column broadcast");
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::BroadcastColumns,rows_,count,nullptr,nullptr,count);
        Tensor out(rows_, count, 0, backend_);
        for (int i = 0; i < rows_; ++i)for (int j = 0; j < count; ++j)out.write()[static_cast<std::size_t>(i) * count +
            j] = read()[i];
        return out;
    }

    Tensor Tensor::broadcast_rows(int count) const
    {
        if (rows_ != 1 || count < 0)throw std::invalid_argument("Tensor row broadcast");
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::BroadcastRows,count,cols_);
        Tensor out(count, cols_, 0, backend_);
        for (int i = 0; i < count; ++i)for (int j = 0; j < cols_; ++j)out.write()[static_cast<std::size_t>(i) * cols_ +
            j] = read()[j];
        return out;
    }

    Tensor Tensor::with_bias_row() const
    {
        if (rows_ == (std::numeric_limits<int>::max)())throw std::length_error("Bias row overflow");
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::BiasRow,rows_+1,cols_);
        Tensor out(rows_ + 1, cols_, 1, backend_);
        if (size())std::copy(read(), read() + size(), out.write() + cols_);
        return out;
    }

    Tensor Tensor::slice_rows(int first, int count) const
    {
        if (first < 0 || count < 0 || first > rows_ || count > rows_ - first)throw std::invalid_argument(
            "Tensor row slice");
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::SliceRows,count,cols_,nullptr,nullptr,0,first);
        Tensor out(count, cols_, 0, backend_);
        if (out.size())std::copy(read() + static_cast<std::size_t>(first) * cols_,
                                 read() + static_cast<std::size_t>(first + count) * cols_, out.write());
        return out;
    }

    Tensor Tensor::zero_first_column() const
    {
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::ZeroFirstColumn,rows_,cols_);
        Tensor out(rows_, cols_, 0, backend_);
        if (size())std::copy(read(), read() + size(), out.write());
        if (cols_)for (int i = 0; i < rows_; ++i)out.write()[static_cast<std::size_t>(i) * cols_] = 0;
        return out;
    }

    Tensor Tensor::activate(Activation activation) const
    {
        if(activation<Activation::Identity||activation>Activation::Softmax||(activation==Activation::Softmax&&rows_==0))throw std::invalid_argument("Invalid activation");
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::Activate,rows_,cols_,nullptr,nullptr,0,static_cast<int>(activation));
        Tensor out(rows_, cols_, 0, backend_);
        if (activation == Activation::Softmax)
        {
            if (rows_ == 0)throw std::invalid_argument("Softmax requires classes");
            for (int j = 0; j < cols_; ++j)
            {
                double m = -std::numeric_limits<double>::infinity(), sum = 0;
                for (int i = 0; i < rows_; ++i)m = std::max(m, read()[static_cast<std::size_t>(i) * cols_ + j]);
                for (int i = 0; i < rows_; ++i)
                {
                    const auto k = static_cast<std::size_t>(i) * cols_ + j;
                    out.write()[k] = std::exp(read()[k] - m);
                    sum += out.read()[k];
                }
                for (int i = 0; i < rows_; ++i)out.write()[static_cast<std::size_t>(i) * cols_ + j] /= sum;
            }
            return out;
        }
        for (std::size_t i = 0; i < size(); ++i)
        {
            const double x = read()[i];
            double y = x;
            switch (activation)
            {
            case Activation::Identity: break;
            case Activation::Sigmoid: y = sigmoid(x);
                break;
            case Activation::Tanh: y = std::tanh(x);
                break;
            case Activation::Relu: y = std::max(0.0, x);
                break;
            case Activation::LeakyRelu: y = x > 0 ? x : 0.01 * x;
                break;
            default: throw std::invalid_argument("Activation");
            }
            out.write()[i] = y;
        }
        return out;
    }

    Tensor Tensor::activation_backward(const Tensor& output, const Tensor& upstream, Activation activation) const
    {
        compatible(output);compatible(upstream);
        if(activation<Activation::Identity||activation>Activation::Softmax)throw std::invalid_argument("Invalid activation");
        if(backend_==Backend::CUDA)return device_operation(gpu::resident::Operation::ActivationBackward,rows_,cols_,&output,&upstream,0,static_cast<int>(activation));
        Tensor out(rows_, cols_, 0, backend_);
        if (activation == Activation::Softmax)
        {
            for (int j = 0; j < cols_; ++j)
            {
                double dot = 0;
                for (int i = 0; i < rows_; ++i)
                {
                    auto k = static_cast<std::size_t>(i) * cols_ + j;
                    dot += output.read()[k] * upstream.read()[k];
                }
                for (int i = 0; i < rows_; ++i)
                {
                    auto k = static_cast<std::size_t>(i) * cols_ + j;
                    out.write()[k] = output.read()[k] * (upstream.read()[k] - dot);
                }
            }
            return out;
        }
        for (std::size_t i = 0; i < size(); ++i)
        {
            const double x = read()[i], a = output.read()[i];
            double d = 1;
            switch (activation)
            {
            case Activation::Identity: break;
            case Activation::Sigmoid: d = a * (1 - a);
                break;
            case Activation::Tanh: d = 1 - a * a;
                break;
            case Activation::Relu: d = x > 0 ? 1 : 0;
                break;
            case Activation::LeakyRelu: d = x > 0 ? 1 : 0.01;
                break;
            default: throw std::invalid_argument("Activation");
            }
            out.write()[i] = upstream.read()[i] * d;
        }
        return out;
    }

    LossResult Tensor::loss(const Tensor& target, LossKind kind) const
    {
        compatible(target);
        if (rows_ == 0 || cols_ == 0)throw std::invalid_argument("Empty loss input");
        if(kind<LossKind::Squared||kind>LossKind::CrossEntropy)throw std::invalid_argument("Unknown loss");
        Tensor values(1, cols_, 0, backend_), gradient(rows_, cols_, 0, backend_);
        if(backend_==Backend::CUDA){
            gpu::resident::loss(buffer_->device_values,target.buffer_->device_values,values.buffer_->device_values,gradient.buffer_->device_values,rows_,cols_,static_cast<int>(kind),buffer_->device);
            if(!values.all_finite()||!gradient.all_finite())throw std::invalid_argument("Invalid or non-finite loss input/result");
            return {values,gradient};
        }
        for (int j = 0; j < cols_; ++j)
        {
            double m = 0, normalizer = 0, target_sum = 0;
            if (kind == LossKind::CrossEntropy)
            {
                m = -std::numeric_limits<double>::infinity();
                for (int i = 0; i < rows_; ++i)m = std::max(m, read()[static_cast<std::size_t>(i) * cols_ + j]);
                for(int i=0;i<rows_;++i){normalizer+=std::exp(read()[static_cast<std::size_t>(i)*cols_+j]-m);target_sum+=target.read()[static_cast<std::size_t>(i)*cols_+j];}
            }
            for (int i = 0; i < rows_; ++i)
            {
                const auto index = static_cast<std::size_t>(i) * cols_ + j;
                const double x = read()[index], y = target.read()[index];
                if (!std::isfinite(x) || !std::isfinite(y))throw std::invalid_argument("Non-finite loss input");
                if (kind != LossKind::Squared && (y < 0 || y > 1))throw std::invalid_argument(
                    "Invalid probability target");
                switch (kind)
                {
                case LossKind::Squared: values.write()[j] += 0.5 * (x - y) * (x - y);
                    gradient.write()[index] = x - y;
                    break;
                case LossKind::BinaryCrossEntropy: values.write()[j] += std::max(x, 0.0) - x * y + std::log1p(
                        std::exp(-std::abs(x)));
                    gradient.write()[index] = sigmoid(x) - y;
                    break;
                case LossKind::CrossEntropy:values.write()[j]+=y*((m-x)+std::log(normalizer));gradient.write()[index]=std::exp(x-m)/normalizer*target_sum-y;break;
                default: throw std::invalid_argument("Unknown loss");
                }
            }
            if (kind == LossKind::CrossEntropy && std::abs(target_sum - 1) > 1e-8)throw std::invalid_argument(
                "Class targets must sum to one");
        }
        if(!values.all_finite()||!gradient.all_finite())throw std::overflow_error("Non-finite loss result");
        return {values,gradient};
    }

    double Tensor::sum()const {if(backend_==Backend::CUDA)return gpu::resident::sum(buffer_->device_values,size(),buffer_->device);double out=0;for(std::size_t i=0;i<size();++i)out+=read()[i];
        return out;}

    double Tensor::item()const {if(size()!=1)throw std::invalid_argument("item requires one element");if(backend_==Backend::CUDA){double value=0;gpu::resident::download(&value,buffer_->device_values,1,buffer_->device);return value;}return read()[0];}
    bool Tensor::all_finite()const {if(backend_==Backend::CUDA)return gpu::resident::all_finite(buffer_->device_values,size(),buffer_->device);for(std::size_t i=0;i<size();++i)if(!std::isfinite(read()[i]))return false;return true;}

    double LossResult::mean() const { return values.sum() / values.cols(); }
    void Tensor::synchronize()const{if(backend_==Backend::CUDA)gpu::resident::synchronize(buffer_->device);}
}
