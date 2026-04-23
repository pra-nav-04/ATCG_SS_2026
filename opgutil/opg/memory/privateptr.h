#pragma once

template <class T>
class PrivatePtr
{
public:
    // Cast to underlying pointer
    inline operator T* () const { return ptr; }

    // Dereference operator
    inline T& operator*() const { return *ptr; };
    // Dereference by accessing member operator
    inline T* operator->() const { return ptr; };

    // Copy constructor
    PrivatePtr(const PrivatePtr &other) = delete;
    PrivatePtr(PrivatePtr &&other) : ptr{ other.ptr } { other.ptr = nullptr; }

    // Assignment operator
    PrivatePtr &operator = (const PrivatePtr &other) = delete;
    PrivatePtr &operator = (PrivatePtr &&other) { ptr = other.ptr; other.ptr = nullptr; }

private:
    friend T;
    // Constructor with raw pointer can only be called from wrapped class itself!
    explicit PrivatePtr(T* _ptr) : ptr{ _ptr } {}

    T* ptr;
};
