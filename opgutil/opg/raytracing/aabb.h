#pragma once

#include <optix.h>
#include <limits>

#include "opg/glmwrapper.h"

#include "opg/preprocessor.h"

#ifndef __CUDACC__
#  include <opg/exception.h>
#  define OPG_AABB_ASSERT(x) OPG_ASSERT(x)
#else
#  define OPG_AABB_ASSERT(x)
#endif


namespace opg {

/**
 * @brief Axis-aligned bounding box
 *
 * @ref Aabb is a utility class for computing and manipulating axis-aligned
 * bounding boxes (aabbs).
 *
 */
struct Aabb
{
public:

    // Construct infinite bounding box
    OPG_HOSTDEVICE Aabb();

    // Construct bounding box that includes a variable number of other things
    template <typename... Args>
    OPG_HOSTDEVICE Aabb(const Args& ...args);

    OPG_HOSTDEVICE bool operator==( const Aabb& other ) const;

    OPG_HOSTDEVICE glm::vec3& operator[]( int i );
    OPG_HOSTDEVICE const glm::vec3& operator[]( int i ) const;

    OPG_HOSTDEVICE operator OptixAabb () const;

    OPG_HOSTDEVICE void reset();

    OPG_HOSTDEVICE bool valid() const;

    OPG_HOSTDEVICE bool contains( const glm::vec3& p ) const;

    OPG_HOSTDEVICE bool contains( const Aabb& bb ) const;

    template <typename T, typename... Args>
    OPG_HOSTDEVICE void include(const T& x, const Args& ...args);

    OPG_HOSTDEVICE glm::vec3 center() const;

    OPG_HOSTDEVICE glm::vec3 extent() const;

    OPG_HOSTDEVICE float volume() const;

    OPG_HOSTDEVICE float area() const;

    OPG_HOSTDEVICE float halfArea() const;

    OPG_HOSTDEVICE int longestAxis() const;

    OPG_HOSTDEVICE float maxExtent() const;

    OPG_HOSTDEVICE bool intersects( const Aabb& other ) const;

    OPG_HOSTDEVICE void intersection( const Aabb& other );

    OPG_HOSTDEVICE void enlarge(float amount);

    OPG_HOSTDEVICE void transform(const glm::mat4& m);

    OPG_HOSTDEVICE bool isFlat() const;

    OPG_HOSTDEVICE float distance( const glm::vec3& x ) const;

    OPG_HOSTDEVICE float distance2( const glm::vec3& x ) const;

    OPG_HOSTDEVICE float signedDistance( const glm::vec3& x ) const;

    glm::vec3 m_min;
    glm::vec3 m_max;
};

OPG_INLINE OPG_HOSTDEVICE Aabb::Aabb()
{
    reset();
}

template <typename... Args>
OPG_INLINE OPG_HOSTDEVICE Aabb::Aabb(const Args& ...args)
{
    reset();
    include(args...);
}

OPG_INLINE OPG_HOSTDEVICE bool Aabb::operator==(const Aabb& other) const
{
    return m_min == other.m_min &&
           m_max == other.m_max;
}

OPG_INLINE OPG_HOSTDEVICE Aabb::operator OptixAabb() const
{
    OptixAabb optixAabb;
    optixAabb.minX = m_min.x;
    optixAabb.minY = m_min.y;
    optixAabb.minZ = m_min.z;
    optixAabb.maxX = m_max.x;
    optixAabb.maxY = m_max.y;
    optixAabb.maxZ = m_max.z;
    return optixAabb;
}

OPG_INLINE OPG_HOSTDEVICE glm::vec3& Aabb::operator[](int i)
{
    OPG_AABB_ASSERT(i >= 0 && i <= 1);
    return i == 0 ? m_min : m_max;
}

OPG_INLINE OPG_HOSTDEVICE const glm::vec3& Aabb::operator[](int i) const
{
    OPG_AABB_ASSERT(i >= 0 && i <= 1);
    return i == 0 ? m_min : m_max;
}

OPG_INLINE OPG_HOSTDEVICE void Aabb::reset()
{
    m_min = glm::vec3(std::numeric_limits<float>::infinity());
    m_max = -glm::vec3(std::numeric_limits<float>::infinity());
}

OPG_INLINE OPG_HOSTDEVICE bool Aabb::valid() const
{
    return glm::all(glm::lessThanEqual(m_min, m_max));
}

OPG_INLINE OPG_HOSTDEVICE bool Aabb::contains(const glm::vec3& p) const
{
    return glm::all(glm::greaterThan(p, m_min)) && glm::all(glm::lessThanEqual(p, m_max));
}

OPG_INLINE OPG_HOSTDEVICE bool Aabb::contains( const Aabb& bb ) const
{
    return glm::all(glm::greaterThan(bb.m_min, m_min)) && glm::all(glm::lessThanEqual(bb.m_max, m_max));
}

template <typename T, typename... Args>
OPG_INLINE OPG_HOSTDEVICE void Aabb::include(const T& x, const Args& ...args)
{
    include(x);
    include(args...);
}

template<>
OPG_INLINE OPG_HOSTDEVICE void Aabb::include(const glm::vec3& p)
{
    m_min = glm::min(m_min, p);
    m_max = glm::max(m_max, p);
}

template<>
OPG_INLINE OPG_HOSTDEVICE void Aabb::include(const Aabb& other)
{
    m_min = glm::min(m_min, other.m_min);
    m_max = glm::max(m_max, other.m_max);
}

OPG_INLINE OPG_HOSTDEVICE glm::vec3 Aabb::center() const
{
    OPG_AABB_ASSERT(valid());
    return (m_min+m_max) * 0.5f;
}

OPG_INLINE OPG_HOSTDEVICE glm::vec3 Aabb::extent() const
{
    OPG_AABB_ASSERT(valid());
    return m_max - m_min;
}

OPG_INLINE OPG_HOSTDEVICE float Aabb::volume() const
{
    OPG_AABB_ASSERT(valid());
    const glm::vec3 d = extent();
    return d.x*d.y*d.z;
}

OPG_INLINE OPG_HOSTDEVICE float Aabb::area() const
{
    return 2.0f * halfArea();
}

OPG_INLINE OPG_HOSTDEVICE float Aabb::halfArea() const
{
    OPG_AABB_ASSERT(valid());
    const glm::vec3 d = extent();
    return d.x*d.y + d.y*d.z + d.z*d.x;
}

OPG_INLINE OPG_HOSTDEVICE int Aabb::longestAxis() const
{
    OPG_AABB_ASSERT(valid());
    const glm::vec3 d = extent();
    if(d.x > d.y)
      return d.x > d.z ? 0 : 2;
    return d.y > d.z ? 1 : 2;
}

OPG_INLINE OPG_HOSTDEVICE float Aabb::maxExtent() const
{
    return extent()[longestAxis()];
}

OPG_INLINE OPG_HOSTDEVICE bool Aabb::intersects(const Aabb& other) const
{
    return !glm::any(glm::greaterThan(other.m_min, m_max) || glm::lessThan(other.m_max, m_min));
}

OPG_INLINE OPG_HOSTDEVICE void Aabb::intersection(const Aabb& other)
{
    m_min = glm::max(m_min, other.m_min);
    m_max = glm::min(m_max, other.m_max);
}

OPG_INLINE OPG_HOSTDEVICE void Aabb::enlarge(float amount)
{
    OPG_AABB_ASSERT(valid());
    m_min -= amount;
    m_max += amount;
}

OPG_INLINE OPG_HOSTDEVICE void Aabb::transform(const glm::mat4& m)
{
    const glm::vec3 corners[] = {
        m_min,
        glm::vec3( m_min.x, m_min.y, m_max.z ),
        glm::vec3( m_min.x, m_max.y, m_min.z ),
        glm::vec3( m_min.x, m_max.y, m_max.z ),
        glm::vec3( m_max.x, m_min.y, m_min.z ),
        glm::vec3( m_max.x, m_min.y, m_max.z ),
        glm::vec3( m_max.x, m_max.y, m_min.z ),
        m_max
    };

    reset();
    for (const auto &corner : corners)
    {
        include(glm::vec3(m * glm::vec4(corner, 1.0f)));
    }
}

OPG_INLINE OPG_HOSTDEVICE bool Aabb::isFlat() const
{
    return m_min == m_max;
}

OPG_INLINE OPG_HOSTDEVICE float Aabb::distance(const glm::vec3& x) const
{
    return std::sqrt(distance2(x));
}

OPG_INLINE OPG_HOSTDEVICE float Aabb::signedDistance(const glm::vec3& x) const
{
    if (glm::all(glm::lessThanEqual(m_min, x)) && glm::all(glm::lessThanEqual(x, m_max)))
    {
        glm::vec3 distance = glm::min(x - m_min, m_max - x);

        float min_distance = glm::min(distance.x, glm::min(distance.y, distance.z));
        return -min_distance;
    }

    return distance(x);
}

OPG_INLINE OPG_HOSTDEVICE float Aabb::distance2(const glm::vec3& x) const
{
    const glm::vec3 ext = extent();

    // compute vector from min corner of box
    const glm::vec3 v = x - m_min;

    float dist2 = 0;
    float excess;

    // project vector from box min to x on each axis,
    // yielding distance to x along that axis, and count
    // any excess distance outside box extents

    excess = 0;
    if (v.x < 0)
        excess = v.x;
    else if (v.x > ext.x)
        excess = v.x - ext.x;
    dist2 += excess * excess;

    excess = 0;
    if (v.y < 0)
        excess = v.y;
    else if (v.y > ext.y)
        excess = v.y - ext.y;
    dist2 += excess * excess;

    excess = 0;
    if (v.z < 0)
        excess = v.z;
    else if (v.z > ext.z)
        excess = v.z - ext.z;
    dist2 += excess * excess;

    return dist2;
}

} // end namespace opg
