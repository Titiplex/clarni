#include "tensor.h"
#include "gpu_backend.h"
#include "tensor_backend.h"
#include <algorithm>
#include <cmath>

// CPU simulation of the resident API. Tests plumbing/lifetimes/transfers only,
// never the CUDA kernel implementation or actual GPU performance.
namespace ml::gpu {
bool compiled()noexcept{return true;}
ProbeResult probe(){return {true,0,"simulated CUDA","mock backend, not hardware"};}
void matmul(const double* a,const double* b,double* c,int rows,int inner,int cols,int){for(int i=0;i<rows;++i)for(int j=0;j<cols;++j){double sum=0;for(int k=0;k<inner;++k)sum+=a[static_cast<std::size_t>(i)*inner+k]*b[static_cast<std::size_t>(k)*cols+j];c[static_cast<std::size_t>(i)*cols+j]=sum;}}
void elementwise(const double* a,const double* b,double* c,std::size_t n,Elementwise operation,double scalar,int){for(std::size_t i=0;i<n;++i)c[i]=operation==Elementwise::Add?a[i]+b[i]:operation==Elementwise::Subtract?a[i]-b[i]:a[i]*scalar;}
}
namespace ml::gpu::resident {
namespace {
Statistics counters;
Tensor tensor(const double* data,int rows,int cols){math::Matrix m(rows,cols);for(int i=0;i<rows;++i)for(int j=0;j<cols;++j)m(i,j)=data[static_cast<std::size_t>(i)*cols+j];return Tensor::from_matrix(m);}
void copy(const Tensor& value,double* output){auto m=value.to_matrix();for(int i=0;i<m.get_x();++i)for(int j=0;j<m.get_y();++j)output[static_cast<std::size_t>(i)*m.get_y()+j]=m(i,j);}
}
double* allocate(std::size_t count,int){if(!count)return nullptr;++counters.allocations;return new double[count];}
void release(double* pointer,int)noexcept{delete[] pointer;}
void fill(double* pointer,std::size_t count,double value,int){if(count)std::fill(pointer,pointer+count,value);}
void upload(double* to,const double* from,std::size_t count,int){if(count){std::copy(from,from+count,to);++counters.uploads;counters.uploaded_bytes+=count*sizeof(double);}}
void download(double* to,const double* from,std::size_t count,int){if(count){std::copy(from,from+count,to);++counters.downloads;counters.downloaded_bytes+=count*sizeof(double);}}
void operation(Operation op,const double* a,const double* b,const double* c,double* out,int rows,int cols,int output_rows,int output_cols,int other,int auxiliary,double scale,double offset,int){
    if(!output_rows||!output_cols)return;
    auto x=tensor(a,rows,cols);Tensor result;
    switch(op){case Operation::Add:result=x+tensor(b,rows,cols);break;case Operation::Subtract:result=x-tensor(b,rows,cols);break;
    case Operation::Multiply:result=x.hadamard(tensor(b,rows,cols));break;case Operation::Divide:result=x.divide(tensor(b,rows,cols));break;
    case Operation::Affine:result=x.affine(scale,offset);break;case Operation::Unary:result=x.unary(static_cast<Unary>(auxiliary));break;
    case Operation::Transpose:result=x.transpose();break;case Operation::Matmul:result=x.matmul(tensor(b,cols,output_cols));break;
    case Operation::SumColumns:result=x.sum_columns();break;case Operation::SumRows:result=x.sum_rows();break;
    case Operation::BroadcastColumns:result=x.broadcast_columns(other);break;case Operation::BroadcastRows:result=x.broadcast_rows(output_rows);break;
    case Operation::BiasRow:result=x.with_bias_row();break;case Operation::SliceRows:result=x.slice_rows(auxiliary,output_rows);break;
    case Operation::ZeroFirstColumn:result=x.zero_first_column();break;case Operation::Activate:result=x.activate(static_cast<Activation>(auxiliary));break;
    case Operation::ActivationBackward:result=x.activation_backward(tensor(b,rows,cols),tensor(c,rows,cols),static_cast<Activation>(auxiliary));break;}
    copy(result,out);
}
void loss(const double* a,const double* b,double* values,double* gradient,int rows,int cols,int kind,int){auto result=tensor(a,rows,cols).loss(tensor(b,rows,cols),static_cast<LossKind>(kind));copy(result.values,values);copy(result.gradient,gradient);}
double sum(const double* pointer,std::size_t count,int){double value=0;for(std::size_t i=0;i<count;++i)value+=pointer[i];return value;}
bool all_finite(const double* pointer,std::size_t count,int){for(std::size_t i=0;i<count;++i)if(!std::isfinite(pointer[i]))return false;return true;}
Statistics statistics()noexcept{return counters;}
void reset_statistics()noexcept{counters={};}
void synchronize(int){}
const char* matmul_implementation()noexcept{return "simulation";}
}
