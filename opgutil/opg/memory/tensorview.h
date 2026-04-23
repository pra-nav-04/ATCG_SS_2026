#pragma once

#include <stdint.h>
#include <type_traits>

#include "opg/preprocessor.h"
#include "opg/memory/bufferview.h"
#include "opg/glmwrapper.h"

namespace opg {

struct Slice
{
    OPG_HOSTDEVICE OPG_INLINE Slice(uint32_t _count) : begin{0}, count{_count}, stride{1} {}
    OPG_HOSTDEVICE OPG_INLINE Slice(uint32_t _begin, uint32_t _count) : begin{_begin}, count{_count}, stride{1} {}
    OPG_HOSTDEVICE OPG_INLINE Slice(uint32_t _begin, uint32_t _count, uint32_t _stride) : begin{_begin}, count{_count}, stride{_stride} {}
    uint32_t begin;
    uint32_t count;
    uint32_t stride;
};

struct Expand
{
    OPG_HOSTDEVICE OPG_INLINE Expand() : count{1} {}
    OPG_HOSTDEVICE OPG_INLINE Expand(uint32_t _count) : count{_count} {}
    uint32_t count;
};

struct Unsqueeze
{
    OPG_HOSTDEVICE OPG_INLINE Unsqueeze() : count{1} {}
    OPG_HOSTDEVICE OPG_INLINE Unsqueeze(uint32_t _count) : count{_count} {}
    uint32_t count;
};

struct Keep
{
    // Nothing.
};

template <typename T, int N>
struct TensorView
{
    static_assert(N > 0, "Dimension count must be non-negative!");

    std::byte*  data    OPG_MEMBER_INIT( nullptr );
    uint32_t    counts[N];
    uint32_t    strides[N];

    TensorView() = default;

    template<typename OT, std::enable_if_t<std::is_convertible<OT(*)[], T(*)[]>::value, bool> = true>
    OPG_INLINE OPG_HOSTDEVICE TensorView(const TensorView<OT, N> &other) :
        data{other.data},
        counts{other.counts},
        strides{other.strides}
    {}

    OPG_INLINE OPG_HOSTDEVICE bool isValid() const
    {
        return static_cast<bool>(data);
    }

    OPG_INLINE OPG_HOSTDEVICE operator bool() const
    {
        return isValid();
    }

    template <int Dim>
    OPG_HOSTDEVICE TensorView<T, N-1> at(uint32_t index) const
    {
        static_assert(Dim >= 0 && Dim < N, "Invalid dimension specified");

        TensorView<T, N-1> result;
        result.data = this->data + this->strides[Dim]*index;
        if constexpr (N > 1)
        {
            for (int i = 0; i < Dim; ++i)
            {
                result.strides[i] = this->strides[i];
                result.counts[i] = this->counts[i];
            }
            for (int i = Dim; i < N-1; ++i)
            {
                result.strides[i] = this->strides[i+1];
                result.counts[i] = this->counts[i+1];
            }
        }
        return result;
    }

    OPG_HOSTDEVICE TensorView<T, N-1> operator[](uint32_t index) const
    {
        return this->at<0>(index);
    }

    template <typename ...Args>
    OPG_HOSTDEVICE auto operator()(Args ...args) const;

    template <int Dim>
    OPG_INLINE OPG_HOSTDEVICE TensorView<T, N> slice(int begin, int count, int stride=1) const
    {
        static_assert(Dim >= 0 && Dim < N, "Invalid dimension specified!");

        if (begin < 0) begin += this->counts[Dim];

        // TODO assert valid begin, end, stride!

        TensorView<T, N> result = *this;
        result.data += result.strides[Dim] * begin;
        result.counts[Dim] = count/stride; // TODO is this the correct behaviour?!!?
        result.strides[Dim] *= stride;
        if (stride < 0)
        {
            result.data -= (result.counts[Dim]-1)*result.strides[Dim];
        }

        return result;
    }

    template <int Dim>
    OPG_INLINE OPG_HOSTDEVICE TensorView<T, N> expand(uint32_t count) const
    {
        static_assert(Dim >= 0 && Dim < N, "Invalid dimension specified!");
        static_assert(this->counts[Dim] == 1, "Invalid count in dimension!");
        TensorView<T, N> result = *this;
        result.strides[Dim] = 0;
        result.counts[Dim] = count;
        return result;
    }

    template <int Dim>
    OPG_INLINE OPG_HOSTDEVICE TensorView<T, N+1> unsqueeze(uint32_t count = 1) const
    {
        // Insert a new "empty" dimension into the tensor view
        static_assert(Dim >= 0 && Dim <= N, "Invalid dimension specified!");
        TensorView<T, N+1> result;
        result.data = this->data;
        for (uint32_t i = 0; i < Dim; ++i)
        {
            result.counts[i] = this->counts[i];
            result.strides[i] = this->strides[i];
        }
        result.counts[Dim] = count;
        result.strides[Dim] = 0;
        for (uint32_t i = Dim; i < N; ++i)
        {
            result.counts[i+1] = this->counts[i];
            result.strides[i+1] = this->strides[i];
        }
        return result;
    }

};

template <typename T>
struct TensorView<T, 0>
{
    std::byte*  data    OPG_MEMBER_INIT( nullptr );

    TensorView() = default;

    template<typename OT, std::enable_if_t<std::is_convertible<OT(*)[], T(*)[]>::value, bool> = true>
    OPG_INLINE OPG_HOSTDEVICE TensorView(const TensorView<OT, 0> &other) :
        data{other.data}
    {}

    OPG_INLINE OPG_HOSTDEVICE bool isValid() const
    {
        return static_cast<bool>(data);
    }

    OPG_INLINE OPG_HOSTDEVICE operator bool() const
    {
        return isValid();
    }

    OPG_INLINE OPG_HOSTDEVICE operator T&() const
    {
        return this->value();
    }

    OPG_HOSTDEVICE void operator = (T value)
    {
        this->value() = value;
    }

    OPG_HOSTDEVICE T* operator -> () const { return reinterpret_cast<T*>(this->data); }

    OPG_INLINE OPG_HOSTDEVICE T& value() const { return *reinterpret_cast<T*>(this->data); }
};


namespace detail {

    template <typename T, int N, int I, typename ...Args>
    struct dispatch_tensor_view_access_operator
    {
        static_assert(sizeof(T) == 0, "Not defined for type!");
    };

    template <typename T, int N, int I, typename U>
    struct dispatch_tensor_view_access_operator<T, N, I, U>
    {
        static constexpr int result_N = N-1;
        static constexpr int result_I = I;
        typedef TensorView<T, result_N> result_type;
        OPG_HOSTDEVICE OPG_INLINE static result_type apply(TensorView<T, N> view, U arg0)
        {
            return view.template at<I>(arg0);
        }
    };

    template <typename T, int N, int I, typename U, glm::qualifier Q>
    struct dispatch_tensor_view_access_operator<T, N, I, glm::vec<1, U, Q>>
    {
        static constexpr int result_N = N-1;
        static constexpr int result_I = I;
        typedef TensorView<T, result_N> result_type;
        OPG_HOSTDEVICE OPG_INLINE static result_type apply(TensorView<T, N> view, glm::vec<1, U, Q> arg0)
        {
            return view.template at<I>(arg0.x);
        }
    };

    template <typename T, int N, int I, typename U, glm::qualifier Q>
    struct dispatch_tensor_view_access_operator<T, N, I, glm::vec<2, U, Q>>
    {
        static constexpr int result_N = N-2;
        static constexpr int result_I = I;
        typedef TensorView<T, result_N> result_type;
        OPG_HOSTDEVICE OPG_INLINE static result_type apply(TensorView<T, N> view, glm::vec<2, U, Q> arg0)
        {
            return view.template at<I>(arg0.y).template at<I>(arg0.x);
        }
    };

    template <typename T, int N, int I, typename U, glm::qualifier Q>
    struct dispatch_tensor_view_access_operator<T, N, I, glm::vec<3, U, Q>>
    {
        static constexpr int result_N = N-3;
        static constexpr int result_I = I;
        typedef TensorView<T, result_N> result_type;
        OPG_HOSTDEVICE OPG_INLINE static result_type apply(TensorView<T, N> view, glm::vec<3, U, Q> arg0)
        {
            return view.template at<I>(arg0.z).template at<I>(arg0.y).template at<I>(arg0.x);
        }
    };

    template <typename T, int N, int I, typename U, glm::qualifier Q>
    struct dispatch_tensor_view_access_operator<T, N, I, glm::vec<4, U, Q>>
    {
        static constexpr int result_N = N-4;
        static constexpr int result_I = I;
        typedef TensorView<T, result_N> result_type;
        OPG_HOSTDEVICE OPG_INLINE static result_type apply(TensorView<T, N> view, glm::vec<4, U, Q> arg0)
        {
            return view.template at<I>(arg0.w).template at<I>(arg0.z).template at<I>(arg0.y).template at<I>(arg0.x);
        }
    };

    template <typename T, int N, int I>
    struct dispatch_tensor_view_access_operator<T, N, I, Slice>
    {
        static constexpr int result_N = N;
        static constexpr int result_I = I+1;
        typedef TensorView<T, result_N> result_type;
        OPG_HOSTDEVICE OPG_INLINE static result_type apply(TensorView<T, N> view, Slice arg0)
        {
            return view.template slice<I>(arg0.begin, arg0.count, arg0.stride);
        }
    };

    template <typename T, int N, int I>
    struct dispatch_tensor_view_access_operator<T, N, I, Expand>
    {
        static constexpr int result_N = N;
        static constexpr int result_I = I+1;
        typedef TensorView<T, result_N> result_type;
        OPG_HOSTDEVICE OPG_INLINE static result_type apply(TensorView<T, N> view, Expand arg0)
        {
            return view.template expand<I>(arg0.count);
        }
    };

    template <typename T, int N, int I>
    struct dispatch_tensor_view_access_operator<T, N, I, Unsqueeze>
    {
        static constexpr int result_N = N+1;
        static constexpr int result_I = I+1;
        typedef TensorView<T, result_N> result_type;
        OPG_HOSTDEVICE OPG_INLINE static result_type apply(TensorView<T, N> view, Unsqueeze arg0)
        {
            return view.template unsqueeze<I>(arg0.count);
        }
    };

    template <typename T, int N, int I>
    struct dispatch_tensor_view_access_operator<T, N, I, Keep>
    {
        static constexpr int result_N = N;
        static constexpr int result_I = I+1;
        typedef TensorView<T, result_N> result_type;
        OPG_HOSTDEVICE OPG_INLINE static result_type apply(TensorView<T, N> view, Keep arg0)
        {
            return view;
        }
    };

    template <typename T, int N, int I, typename U, typename ...Args>
    struct dispatch_tensor_view_access_operator<T, N, I, U, Args...>
    {
        static constexpr int temp_N = dispatch_tensor_view_access_operator<T, N, I, U>::result_N;
        static constexpr int temp_I = dispatch_tensor_view_access_operator<T, N, I, U>::result_I;
        static constexpr int result_N = dispatch_tensor_view_access_operator<T, temp_N, temp_I, Args...>::result_N;
        static constexpr int result_I = dispatch_tensor_view_access_operator<T, temp_N, temp_I, Args...>::result_I;
        typedef typename dispatch_tensor_view_access_operator<T, temp_N, temp_I, Args...>::result_type result_type;

        OPG_HOSTDEVICE OPG_INLINE static result_type apply(TensorView<T, N> view, U arg0, Args ...args)
        {
            auto temp = dispatch_tensor_view_access_operator<T, N, I, U>::apply(view, arg0);
            auto result = dispatch_tensor_view_access_operator<T, temp_N, temp_I, Args...>::apply(temp, args...);
            return result;
        }
    };

} // namespace detail

template <typename T, int N>
template <typename... Args>
auto TensorView<T, N>::operator () (Args ...args) const
{
    return detail::dispatch_tensor_view_access_operator<T, N, 0, Args...>::apply(*this, args...);
}


namespace detail {

    template <typename T, int N, int I, typename ...Args>
    struct set_counts_and_strides
    {
        static_assert(sizeof(T) == 0, "Not defined for type!");
    };

    template <typename T, int N, int I, typename Arg, typename ...Args>
    struct set_counts_and_strides<T, N, I, Arg, Args...>
    {
        static uint32_t apply(TensorView<T, N> &view, uint32_t base_stride, Arg arg, Args ...args)
        {
            uint32_t stride = set_counts_and_strides<T, N, I+1, Args...>::apply(view, base_stride, args...);

            view.counts[I] = arg;
            view.strides[I] = stride;
            return stride * arg;
        }
    };

    template <typename T, int N>
    struct set_counts_and_strides<T, N, N>
    {
        static uint32_t apply(TensorView<T, N> &view, uint32_t base_stride)
        {
            return base_stride;
        }
    };

} // namespace detail



template <typename T, typename ...Args>
TensorView<T, sizeof...(Args)> make_tensor_view(T *data, uint32_t base_stride, Args ...args)
{
    TensorView<T, sizeof...(Args)> result;
    result.data = reinterpret_cast<std::byte*>(data);
    detail::set_counts_and_strides<T, sizeof...(Args), 0, Args...>::apply(result, base_stride, args...);
    return result;
}

template <typename T, typename ...Args>
TensorView<T, sizeof...(Args)> make_tensor_view(const BufferView<T> &buffer_view, Args ...args)
{
    TensorView<T, sizeof...(Args)> result;
    result.data = buffer_view.data;
    detail::set_counts_and_strides<T, sizeof...(Args), 0, Args...>::apply(result, buffer_view.byte_stride, args...);
    return result;
}

template <typename T>
OPG_INLINE OPG_HOSTDEVICE BufferView<T> make_buffer_view(const TensorView<T, 1> &tensor_view)
{
    // Convert a 1D tensor view into a buffer view.
    BufferView<T> buffer_view;
    buffer_view.data = tensor_view.data;
    buffer_view.count = tensor_view.counts[0];
    buffer_view.byte_stride = tensor_view.strides[0];
    buffer_view.elmt_byte_size = sizeof(T);
    return buffer_view;
}

template <typename T, int N>
OPG_INLINE OPG_HOSTDEVICE TensorView<T, N> expand_tensor_view(TensorView<T, N> view, uint32_t dim, uint32_t size)
{
    //OPG_ASSERT(view.counts[dim] == 1);
    view.strides[dim] = 0;
    view.counts[dim] = size;
    return view;
}

template <typename T, int N>
OPG_INLINE OPG_HOSTDEVICE TensorView<T, N+1> unsqueeze_tensor_view(TensorView<T, N> view, uint32_t dim, uint32_t size = 1)
{
    // Insert a new "empty" dimension into the tensor view
    //OPG_ASSERT(dim <= N);
    TensorView<T, N+1> output_view;
    output_view.data = view.data;
    for (uint32_t i = 0; i < dim; ++i)
    {
        output_view.counts[i] = view.counts[i];
        output_view.strides[i] = view.strides[i];
    }
    output_view.counts[dim] = size;
    output_view.strides[dim] = 0;
    for (uint32_t i = dim; i < N; ++i)
    {
        output_view.counts[i+1] = view.counts[i];
        output_view.strides[i+1] = view.strides[i];
    }
    return output_view;
}

} // namespace opg
