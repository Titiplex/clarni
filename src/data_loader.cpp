#include "data_loader.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
namespace ml::data
{
    std::size_t Batch::valid_targets(int ignore) const
    {
        return static_cast<std::size_t>(
            std::count_if(targets.begin(), targets.end(), [=](int y) { return y != ignore; }));
    }
    Subset::Subset(std::shared_ptr<const Dataset> source, std::vector<std::size_t> indices)
        : source_(std::move(source)), indices_(std::move(indices))
    {
        if (!source_)
            throw std::invalid_argument("Null subset dataset");
        for (auto i : indices_)
            if (i >= source_->size())
                throw std::out_of_range("Subset index");
    }
    Split random_split(std::size_t size, std::size_t validation, std::size_t test,
                       std::uint32_t seed)
    {
        if (validation > size || test > size - validation)
            throw std::invalid_argument("Invalid split sizes");
        std::vector<std::size_t> order(size);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(order.begin(), order.end(), rng);
        auto train_end = order.begin() + static_cast<std::ptrdiff_t>(size - validation - test),
             valid_end = train_end + static_cast<std::ptrdiff_t>(validation);
        return {{order.begin(), train_end}, {train_end, valid_end}, {valid_end, order.end()}};
    }
    Collator make_collator(CollateOptions opts)
    {
        if (opts.input != InputLayout::Stack && opts.input != InputLayout::PadTokens &&
            opts.input != InputLayout::PadFeatures)
            throw std::invalid_argument("Unknown input layout");
        if (opts.targets != TargetLayout::Class && opts.targets != TargetLayout::Sequence)
            throw std::invalid_argument("Unknown target layout");
        if (opts.input == InputLayout::Stack && opts.targets != TargetLayout::Class)
            throw std::invalid_argument("Stack collator requires class targets");
        if (opts.padding_id < 0)
            throw std::invalid_argument("Negative padding ID");
        return [opts](const std::vector<Example> &examples,
                      const std::vector<std::size_t> &indices) -> Batch
        {
            if (examples.empty() || examples.size() != indices.size() ||
                examples.size() > std::size_t((std::numeric_limits<int>::max)()))
                throw std::invalid_argument("Invalid collate batch");
            const int batch = static_cast<int>(examples.size());
            if (opts.input == InputLayout::Stack)
            {
                if (!std::holds_alternative<nd::Array>(examples[0].input))
                    throw std::invalid_argument("Stack expects arrays");
                auto sample_shape = std::get<nd::Array>(examples[0].input).shape(),
                     shape = sample_shape;
                shape.insert(shape.begin(), batch);
                nd::Vector values;
                values.reserve(nd::numel(shape));
                std::vector<int> targets;
                for (auto &e : examples)
                {
                    if (!std::holds_alternative<nd::Array>(e.input) ||
                        std::get<nd::Array>(e.input).shape() != sample_shape ||
                        e.targets.size() != 1)
                        throw std::invalid_argument("Stack sample shape/label mismatch");
                    auto v = std::get<nd::Array>(e.input).values();
                    values.insert(values.end(), v.begin(), v.end());
                    targets.push_back(e.targets[0]);
                }
                return {nd::Array::from(shape, values), targets, {}, indices};
            }
            std::vector<int> lengths;
            int maximum = 0;
            nd::Shape tail;
            for (auto &e : examples)
            {
                int length;
                if (opts.input == InputLayout::PadTokens)
                {
                    if (!std::holds_alternative<std::vector<int>>(e.input))
                        throw std::invalid_argument("Token collator expects integer sequences");
                    auto &ids = std::get<std::vector<int>>(e.input);
                    if (ids.empty() || ids.size() > std::size_t((std::numeric_limits<int>::max)()))
                        throw std::invalid_argument("Invalid token sequence length");
                    for (int id : ids)
                        if (id < 0)
                            throw std::invalid_argument("Negative token ID");
                    length = static_cast<int>(ids.size());
                }
                else
                {
                    if (!std::holds_alternative<nd::Array>(e.input))
                        throw std::invalid_argument("Feature collator expects arrays");
                    auto shape = std::get<nd::Array>(e.input).shape();
                    if (shape.size() < 2)
                        throw std::invalid_argument("Feature sequence requires [T,...features]");
                    length = shape[0];
                    shape.erase(shape.begin());
                    if (tail.empty())
                        tail = shape;
                    else if (tail != shape)
                        throw std::invalid_argument("Sequence feature shape mismatch");
                }
                if (e.targets.size() !=
                    (opts.targets == TargetLayout::Class ? 1 : static_cast<std::size_t>(length)))
                    throw std::invalid_argument("Target sequence length mismatch");
                maximum = (std::max)(maximum, length);
                lengths.push_back(length);
            }
            nd::Shape shape{batch, maximum};
            nd::numel(shape);
            std::vector<int> targets(
                opts.targets == TargetLayout::Class ? batch : std::size_t(batch) * maximum,
                opts.ignore_index);
            for (int b = 0; b < batch; ++b)
            {
                if (opts.targets == TargetLayout::Class)
                    targets[b] = examples[b].targets[0];
                else
                    std::copy(examples[b].targets.begin(), examples[b].targets.end(),
                              targets.begin() + std::size_t(b) * maximum);
            }
            if (opts.input == InputLayout::PadTokens)
            {
                std::vector<int> values(std::size_t(batch) * maximum, opts.padding_id);
                for (int b = 0; b < batch; ++b)
                {
                    auto &v = std::get<std::vector<int>>(examples[b].input);
                    std::copy(v.begin(), v.end(), values.begin() + std::size_t(b) * maximum);
                }
                return {nn::TokenBatch{values, shape}, targets, lengths, indices};
            }
            shape.insert(shape.end(), tail.begin(), tail.end());
            nd::Vector values(nd::numel(shape), 0);
            auto features = nd::numel(tail);
            for (int b = 0; b < batch; ++b)
            {
                auto v = std::get<nd::Array>(examples[b].input).values();
                std::copy(v.begin(), v.end(), values.begin() + std::size_t(b) * maximum * features);
            }
            return {nd::Array::from(shape, values), targets, lengths, indices};
        };
    }
    DataLoader::DataLoader(std::shared_ptr<const Dataset> data, LoaderOptions options,
                           Collator collator)
        : dataset_(std::move(data)), options_(std::move(options)), collate_(std::move(collator)),
          generator_(options_.seed), cursor_(0)
    {
        if (!dataset_ || !options_.batch_size || !collate_)
            throw std::invalid_argument("Invalid DataLoader options");
        order_.resize(dataset_->size());
        std::iota(order_.begin(), order_.end(), 0);
        cursor_ = order_.size();
    }
    void DataLoader::reset_epoch()
    {
        if (epoch_ == (std::numeric_limits<std::size_t>::max)())
            throw std::overflow_error("Epoch counter overflow");
        std::iota(order_.begin(), order_.end(), 0);
        if (options_.shuffle)
            std::shuffle(order_.begin(), order_.end(), generator_);
        cursor_ = 0;
        ++epoch_;
    }
    std::optional<Batch> DataLoader::next()
    {
        if (cursor_ == order_.size())
            return std::nullopt;
        std::size_t n = (std::min)(options_.batch_size, order_.size() - cursor_);
        if (options_.drop_last && n < options_.batch_size)
        {
            cursor_ = order_.size();
            return std::nullopt;
        }
        std::vector<Example> examples;
        std::vector<std::size_t> indices;
        examples.reserve(n);
        indices.reserve(n);
        for (std::size_t j = 0; j < n; ++j)
        {
            auto index = order_[cursor_ + j];
            examples.push_back(dataset_->get(index));
            indices.push_back(index);
        }
        auto result = collate_(examples, indices);
        cursor_ += n;
        return result;
    }
    LoaderState DataLoader::state() const
    {
        std::ostringstream out;
        out << generator_;
        return {order_,           cursor_,
                epoch_,           options_.batch_size,
                options_.shuffle, options_.drop_last,
                out.str(),        options_.dataset_key};
    }
    void DataLoader::restore(const LoaderState &s)
    {
        if (s.order.size() != order_.size() || s.cursor > s.order.size() ||
            s.batch_size != options_.batch_size || s.shuffle != options_.shuffle ||
            s.drop_last != options_.drop_last || s.dataset_key != options_.dataset_key)
            throw std::invalid_argument("Loader checkpoint dataset/options mismatch");
        if ((!s.epoch && s.cursor != s.order.size()) ||
            (s.cursor != s.order.size() && s.cursor % options_.batch_size))
            throw std::invalid_argument("Invalid loader cursor");
        std::vector<bool> seen(s.order.size(), false);
        for (auto i : s.order)
        {
            if (i >= seen.size() || seen[i])
                throw std::invalid_argument("Loader order is not a permutation");
            seen[i] = true;
        }
        std::mt19937 rng;
        std::istringstream in(s.rng);
        if (!(in >> rng))
            throw std::invalid_argument("Invalid loader RNG");
        in >> std::ws;
        if (!in.eof())
            throw std::invalid_argument("Trailing loader RNG");
        auto order = s.order;
        order_.swap(order);
        cursor_ = s.cursor;
        epoch_ = s.epoch;
        generator_ = rng;
    }
    void DataLoader::save_state(const std::string &path) const
    {
        auto s = state();
        std::ofstream out(path);
        if (!out)
            throw std::runtime_error("Cannot write loader state");
        out << "CLARNI_LOADER 1\n"
            << s.order.size() << ' ' << s.cursor << ' ' << s.epoch << ' ' << s.batch_size << ' '
            << s.shuffle << ' ' << s.drop_last << '\n'
            << std::quoted(s.dataset_key) << '\n'
            << std::quoted(s.rng) << '\n';
        for (auto i : s.order)
            out << i << ' ';
        out << '\n';
        if (!out)
            throw std::runtime_error("Loader state write failed");
    }
    void DataLoader::load_state(const std::string &path)
    {
        std::ifstream in(path);
        std::string magic;
        int version, shuffle, drop;
        std::size_t size;
        LoaderState s;
        if (!(in >> magic >> version >> size >> s.cursor >> s.epoch >> s.batch_size >> shuffle >>
              drop) ||
            magic != "CLARNI_LOADER" || version != 1 || size != order_.size() || shuffle < 0 ||
            shuffle > 1 || drop < 0 || drop > 1)
            throw std::invalid_argument("Invalid loader state header");
        s.shuffle = shuffle;
        s.drop_last = drop;
        if (!(in >> std::quoted(s.dataset_key) >> std::quoted(s.rng)))
            throw std::invalid_argument("Invalid loader state metadata");
        s.order.resize(size);
        for (auto &i : s.order)
            if (!(in >> i))
                throw std::invalid_argument("Truncated loader order");
        in >> std::ws;
        if (!in.eof())
            throw std::invalid_argument("Trailing loader state");
        restore(s);
    }
    namespace
    {
        std::uint32_t read_u32(std::istream &in)
        {
            unsigned char bytes[4];
            if (!in.read(reinterpret_cast<char *>(bytes), 4))
                throw std::invalid_argument("Truncated IDX header");
            return (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16) |
                   (std::uint32_t(bytes[2]) << 8) | bytes[3];
        }
    } // namespace
    MNIST::MNIST(const std::string &images, const std::string &labels, std::size_t limit)
    {
        std::ifstream image_stream(images, std::ios::binary),
            label_stream(labels, std::ios::binary);
        if (!image_stream || !label_stream)
            throw std::runtime_error("Cannot open raw MNIST IDX files");
        auto image_magic = read_u32(image_stream), count = read_u32(image_stream),
             rows = read_u32(image_stream), cols = read_u32(image_stream);
        auto label_magic = read_u32(label_stream), label_count = read_u32(label_stream);
        if (image_magic != 2051 || label_magic != 2049 || count != label_count || !count || !rows ||
            !cols || rows > std::uint32_t((std::numeric_limits<int>::max)()) ||
            cols > std::uint32_t((std::numeric_limits<int>::max)()))
            throw std::invalid_argument("Invalid MNIST IDX magic/count/shape");
        rows_ = static_cast<int>(rows);
        cols_ = static_cast<int>(cols);
        auto image_size = nd::numel({rows_, cols_});
        if (count > ((std::numeric_limits<std::uintmax_t>::max)() - 16) / image_size ||
            std::filesystem::file_size(images) != 16 + std::uintmax_t(count) * image_size ||
            std::filesystem::file_size(labels) != 8 + std::uintmax_t(count))
            throw std::invalid_argument("MNIST IDX file size mismatch");
        std::size_t n = limit ? (std::min)(limit, std::size_t(count)) : count;
        if (n > pixels_.max_size() / image_size ||
            n * image_size > std::size_t((std::numeric_limits<std::streamsize>::max)()))
            throw std::length_error("MNIST dataset too large");
        pixels_.resize(n * image_size);
        labels_.resize(n);
        if (!image_stream.read(reinterpret_cast<char *>(pixels_.data()),
                               static_cast<std::streamsize>(pixels_.size())) ||
            !label_stream.read(reinterpret_cast<char *>(labels_.data()),
                               static_cast<std::streamsize>(labels_.size())))
            throw std::invalid_argument("Truncated MNIST dataset");
        for (unsigned char y : labels_)
            if (y > 9)
                throw std::invalid_argument("MNIST label outside 0..9");
    }
    Example MNIST::get(std::size_t i) const
    {
        auto label = labels_.at(i);
        auto count = std::size_t(rows_) * cols_;
        nd::Vector values(count);
        for (std::size_t j = 0; j < count; ++j)
            values[j] = pixels_[i * count + j] / 255.0;
        return {nd::Array::from({1, rows_, cols_}, values), {label}};
    }
} // namespace ml::data
