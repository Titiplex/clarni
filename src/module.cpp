#include "module.h"
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace ml::nn
{
    Parameter::Parameter(const nd::Array& a) : value_(nd::Array::from(a.shape(), a.values(), true))
    {
        if (!value_.variable().value().all_finite())
            throw std::invalid_argument("Non-finite parameter");
    }

    void Parameter::assign(const nd::Array& a)
    {
        if (a.shape() != value_.shape() || !a.variable().value().all_finite())
            throw std::invalid_argument("Invalid parameter replacement");
        value_ = nd::Array(a.shape(), ad::Variable(a.variable().value(), true));
    }

    void Parameter::zero_grad()
    {
        value_ = nd::Array(value_.shape(), ad::Variable(value_.variable().value(), true));
    }

    nd::Array Module::forward(const nd::Array&)
    {
        throw std::logic_error("Use module typed forward interface");
    }

    std::vector<NamedParameter> parameters(const Module& m)
    {
        std::vector<NamedParameter> out;
        std::unordered_set<Parameter*> seen;
        std::unordered_set<std::string> names;
        for (auto p : m.named_parameters())
        {
            if (!p.parameter || p.name.empty() || !names.insert(p.name).second)
                throw std::invalid_argument("Invalid parameter names");
            if (seen.insert(p.parameter.get()).second)
                out.push_back(std::move(p));
        }
        return out;
    }

    Param normal_parameter(nd::Shape shape, double stddev, std::mt19937& gen)
    {
        if (!std::isfinite(stddev) || stddev <= 0)
            throw std::invalid_argument("Invalid initializer");
        nd::Vector v(nd::numel(shape));
        std::normal_distribution<double> d(0, stddev);
        for (auto& x : v)
            x = d(gen);
        return std::make_shared<Parameter>(nd::Array::from(std::move(shape), v));
    }

    Linear::Linear(int inputs, int outputs, const std::uint32_t seed) : in_(inputs), out_(outputs)
    {
        nd::numel({inputs, outputs});
        std::mt19937 gen(seed);
        weight = normal_parameter({inputs, outputs}, std::sqrt(2.0 / (static_cast<double>(inputs) + outputs)), gen);
        bias = std::make_shared<Parameter>(nd::Array({outputs}));
    }

    nd::Array Linear::forward(const nd::Array& a)
    {
        if (a.shape().empty() || a.shape().back() != in_)
            throw std::invalid_argument("Linear feature mismatch");
        auto shape = a.shape();
        shape.back() = out_;
        return nd::reshape(
            nd::add(nd::matmul(nd::reshape(a, {static_cast<int>(a.size() / in_), in_}), weight->value()),
                    bias->value()),
            shape);
    }

    std::vector<NamedParameter> Linear::named_parameters() const
    {
        return {{"weight", weight}, {"bias", bias}};
    }

    nd::Array Flatten::forward(const nd::Array& a)
    {
        if (a.shape().empty())
            throw std::invalid_argument("Flatten requires batch axis");
        return nd::reshape(a, {a.shape()[0], static_cast<int>(a.size() / a.shape()[0])});
    }

    void Sequential::add(std::shared_ptr<Module> m)
    {
        if (!m || m.get() == this)
            throw std::invalid_argument("Invalid child");
        m->train(training());
        modules_.push_back(std::move(m));
    }

    nd::Array Sequential::forward(const nd::Array& a)
    {
        auto out = a;
        for (const auto& m : modules_)
            out = m->forward(out);
        return out;
    }

    std::vector<NamedParameter> Sequential::named_parameters() const
    {
        std::vector<NamedParameter> out;
        for (std::size_t i = 0; i < modules_.size(); ++i)
            for (auto p : modules_[i]->named_parameters())
            {
                p.name = std::to_string(i) + "." + p.name;
                out.push_back(std::move(p));
            }
        return out;
    }

    void Sequential::train(const bool enabled)
    {
        Module::train(enabled);
        for (const auto& m : modules_)
            m->train(enabled);
    }

    Trainer::Trainer(Module& m, const OptimizerOptions& options)
        : module_(m), bindings_(parameters(m)), optimizer_(options)
    {
        if (bindings_.empty())
            throw std::invalid_argument("No trainable parameters");
    }

    void Trainer::validate_bindings() const
    {
        auto p = parameters(module_);
        if (p.size() != bindings_.size())
            throw std::logic_error("Trainer topology changed");
        for (std::size_t i = 0; i < p.size(); ++i)
            if (p[i].name != bindings_[i].name || p[i].parameter != bindings_[i].parameter)
                throw std::logic_error("Trainer parameter identities changed");
    }

    void Trainer::validate_model(const Module& model) const
    {
        if (&model != &module_)
            throw std::invalid_argument("Trainer belongs to a different model");
        validate_bindings();
    }

    void Trainer::zero_grad() const
    {
        validate_bindings();
        for (auto& p : bindings_)
            p.parameter->zero_grad();
    }

    double Trainer::grad_norm() const
    {
        validate_bindings();
        double n = 0;
        for (auto& p : bindings_)
            if (p.parameter->value().has_grad())
                for (double g : p.parameter->value().grad())
                {
                    if (!std::isfinite(g))
                        throw std::overflow_error("Non-finite gradient");
                    n = std::hypot(n, g);
                }
        if (!std::isfinite(n))
            throw std::overflow_error("Gradient norm overflow");
        return n;
    }

    void Trainer::step(const double rate, double max_grad_norm)
    {
        validate_bindings();
        if (!std::isfinite(rate) || rate <= 0 || !std::isfinite(max_grad_norm) || max_grad_norm < 0)
            throw std::invalid_argument("Invalid step");
        const double n = grad_norm();
        const double f = max_grad_norm > 0 && n > max_grad_norm ? max_grad_norm / n : 1;
        auto opt = optimizer_;
        std::vector<std::pair<Param, nd::Array>> pending;
        for (std::size_t i = 0; i < bindings_.size(); ++i)
        {
            auto p = bindings_[i].parameter;
            auto& a = p->value();
            if (!a.has_grad())
                continue;
            const auto next = opt.update(i, a.variable().value(), a.variable().grad().affine(f), rate);
            pending.emplace_back(p, nd::Array(a.shape(), ad::Variable(next, false)));
        }
        if (pending.empty())
            throw std::logic_error("Run backward before step");
        for (auto& [p, a] : pending)
            p->assign(a);
        optimizer_ = std::move(opt);
    }

    void Trainer::restore_optimizer(const std::vector<Optimizer::State>& s)
    {
        validate_bindings();
        if (s.size() > bindings_.size())
            throw std::invalid_argument("Too many optimizer states");
        for (std::size_t i = 0; i < s.size(); ++i)
            if (s[i].steps &&
                (s[i].rows != static_cast<int>(bindings_[i].parameter->value().size()) || s[i].cols != 1))
                throw std::invalid_argument("Optimizer shape mismatch");
        optimizer_.restore_states(s);
    }

    namespace
    {
        void write_tensor(std::ostream& o, const Tensor& t)
        {
            auto m = t.to_matrix();
            for (int i = 0; i < m.get_x(); ++i)
                for (int j = 0; j < m.get_y(); ++j)
                    o << m(i, j) << ' ';
            o << '\n';
        }

        Tensor read_tensor(std::istream& in, const int r, const int c)
        {
            math::Matrix m(r, c);
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j)
                {
                    if (!(in >> m(i, j)) || !std::isfinite(m(i, j)))
                        throw std::invalid_argument("Invalid checkpoint tensor");
                }
            return Tensor::from_matrix(m);
        }
    } // namespace
    void save_checkpoint(const std::string& path, const Module& m, const Trainer* trainer)
    {
        if (trainer)
            trainer->validate_model(m);
        auto p = parameters(m);
        std::ofstream out(path);
        if (!out)
            throw std::runtime_error("Cannot write checkpoint");
        out << std::setprecision(std::numeric_limits<double>::max_digits10) << "CLARNI_NN 1\n"
            << p.size() << '\n';
        for (auto& n : p)
        {
            auto& a = n.parameter->value();
            out << std::quoted(n.name) << ' ' << a.shape().size();
            for (const int d : a.shape())
                out << ' ' << d;
            out << '\n';
            write_tensor(out, a.variable().value());
        }
        out << (trainer ? 1 : 0) << '\n';
        if (trainer)
        {
            auto& opt = trainer->optimizer();
            auto& o = opt.options();
            out << static_cast<int>(o.kind) << ' ' << o.momentum << ' ' << o.beta1 << ' ' << o.beta2 << ' '
                << o.epsilon << '\n'
                << opt.states().size() << '\n';
            for (auto& s : opt.states())
            {
                out << s.steps << ' ' << s.rows << ' ' << s.cols << '\n';
                if (s.steps && o.kind != OptimizerKind::SGD)
                    write_tensor(out, s.first);
                if (s.steps && o.kind == OptimizerKind::Adam)
                    write_tensor(out, s.second);
            }
        }
        if (!out)
            throw std::runtime_error("Checkpoint write failed");
    }

    void load_checkpoint(const std::string& path, Module& m, Trainer* trainer)
    {
        if (trainer)
            trainer->validate_model(m);
        std::ifstream in(path);
        std::string magic;
        int version;
        std::size_t count;
        auto params = parameters(m);
        if (!(in >> magic >> version >> count) || magic != "CLARNI_NN" || version != 1 ||
            count != params.size())
            throw std::invalid_argument("Invalid checkpoint header/topology");
        std::vector<nd::Array> pending;
        for (auto& p : params)
        {
            std::string name;
            std::size_t rank;
            if (!(in >> std::quoted(name) >> rank) || name != p.name ||
                rank != p.parameter->value().shape().size())
                throw std::invalid_argument("Checkpoint name/rank mismatch");
            nd::Shape shape(rank);
            for (std::size_t j = 0; j < rank; ++j)
                if (!(in >> shape[j]) || shape[j] != p.parameter->value().shape()[j])
                    throw std::invalid_argument("Checkpoint shape mismatch");
            pending.emplace_back(shape,
                                 ad::Variable(read_tensor(in, static_cast<int>(nd::numel(shape)), 1), false));
        }
        int has;
        std::vector<Optimizer::State> states;
        if (!(in >> has) || (has != 0 && has != 1))
            throw std::invalid_argument("Invalid optimizer flag");
        if (has)
        {
            int kind;
            OptimizerOptions o;
            if (!(in >> kind >> o.momentum >> o.beta1 >> o.beta2 >> o.epsilon) || kind < 0 ||
                kind > 2)
                throw std::invalid_argument("Invalid optimizer header");
            o.kind = static_cast<OptimizerKind>(kind);
            Optimizer candidate(o);
            if (trainer)
            {
                auto e = trainer->optimizer().options();
                if (o.kind != e.kind || o.momentum != e.momentum || o.beta1 != e.beta1 ||
                    o.beta2 != e.beta2 || o.epsilon != e.epsilon)
                    throw std::invalid_argument("Optimizer options mismatch");
            }
            std::size_t n;
            if (!(in >> n) || n > params.size())
                throw std::invalid_argument("Invalid state count");
            states.resize(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                auto& s = states[i];
                if (!(in >> s.steps >> s.rows >> s.cols))
                    throw std::invalid_argument("Invalid state");
                if (!s.steps)
                {
                    if (s.rows != 0 || s.cols != 0)
                        throw std::invalid_argument("Invalid unused state");
                    continue;
                }
                if (s.rows != static_cast<int>(params[i].parameter->value().size()) || s.cols != 1)
                    throw std::invalid_argument("Optimizer shape mismatch");
                if (o.kind != OptimizerKind::SGD)
                    s.first = read_tensor(in, s.rows, s.cols);
                if (o.kind == OptimizerKind::Adam)
                    s.second = read_tensor(in, s.rows, s.cols);
            }
            candidate.restore_states(states);
        }
        in >> std::ws;
        if (!in.eof())
            throw std::invalid_argument("Trailing checkpoint content");
        if (trainer)
            trainer->restore_optimizer(states);
        for (std::size_t i = 0; i < params.size(); ++i)
            params[i].parameter->assign(pending[i]);
    }
} // namespace ml::nn
