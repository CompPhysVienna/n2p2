// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef VEC3D_H
#define VEC3D_H

#include <cstddef>   // std::size_t
#include <cmath>     // sqrt
#include <stdexcept> // std::runtime_error

namespace nnp
{

/// Vector in 3 dimensional real space.
struct Vec3D
{
    /// cartesian coordinates.
    double r[3];

    /** Constructor, initializes to zero.
     */
    Vec3D();
    /** Constructor, with initialization of coordinates.
     */
    Vec3D(double x, double y, double z);
    /** Copy constructor.
     */
    Vec3D(Vec3D const& source);
    /** Overload = operator.
     *
     * @return Vector with copied data.
     */
    Vec3D&        operator=(Vec3D const& rhs);
    /** Overload += operator to implement in-place vector addition.
     *
     * @return Replace original vector with sum of two vectors.
     */
    Vec3D&        operator+=(Vec3D const& v);
    /** Overload -= operator to implement in-place vector subtraction.
     *
     * @return Replace original vector with original minus given.
     */
    Vec3D&        operator-=(Vec3D const& v);
    /** Overload *= operator to implement multiplication with scalar.
     *
     * @return Original vector multiplied with scalar.
     */
    Vec3D&        operator*=(double const a);
    /** Overload /= operator to implement division by scalar.
     *
     * @return Original vector divided by scalar.
     */
    Vec3D&        operator/=(double const a);
    /** Overload * operator to implement scalar product.
     *
     * @return Scalar product of two vectors.
     */
    double        operator*(Vec3D const& v) const;
    /** Overload [] operator to return coordinate by index.
     *
     * @return x,y or z when index is 0, 1 or 2, respectively.
     */
    double&       operator[](std::size_t const index);
    /** Overload [] operator to return coordinate by index (const version).
     *
     * @return x,y or z when index is 0, 1 or 2, respectively.
     */
    double const& operator[](std::size_t const index) const;
    /** Compare if vectors are equal.
     *
     * @return `True` if all components are equal, `false` else.
     */
    bool          operator==(Vec3D const& rhs) const;
    /** Compare if vectors are not equal.
     *
     * @return `False` if all components are equal, `false` else.
     */
    bool          operator!=(Vec3D const& rhs) const;
    /** Calculate norm of vector.
     *
     * @return Norm of vector.
     */
    double        norm() const;
    /** Calculate square of norm of vector.
     *
     * @return Square of norm of vector.
     */
    double        norm2() const;
    /** Calculate l1 norm of vector (taxicab metric).
     *
     * @return L1-norm of vector.
     */
    double        l1norm() const;
    /** Normalize vector, norm equals 1.0 afterwards.
     */
    Vec3D&        normalize();
    /** Cross product, argument vector is second in product.
     *
     * @return Cross product of two vectors.
     */
    Vec3D         cross(Vec3D const& v) const;
};

/** Overload + operator to implement vector addition.
 *
 * @return Sum of two vectors.
 */
Vec3D operator+(Vec3D lhs, Vec3D const& rhs);
/** Overload - operator to implement vector subtraction.
 *
 * @return Difference between two vectors.
 */
Vec3D operator-(Vec3D lhs, Vec3D const& rhs);
/** Overload - operator to implement vector sign change.
 *
 * @return Negative vector.
 */
Vec3D operator-(Vec3D v);
/** Overload * operator to implement multiplication with scalar.
 *
 * @return Vector multiplied with scalar.
 */
Vec3D operator*(Vec3D v, double const a);
/** Overload / operator to implement division by scalar.
 *
 * @return Vector divided by scalar.
 */
Vec3D operator/(Vec3D v, double const a);
/** Overload * operator to implement multiplication with scalar (scalar first).
 *
 * @return Vector multiplied with scalar.
 */
Vec3D operator*(double const a, Vec3D v);

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline Vec3D::Vec3D()
{
    r[0] = 0.0;
    r[1] = 0.0;
    r[2] = 0.0;
}

inline Vec3D::Vec3D(double x, double y, double z)
{
    r[0] = x;
    r[1] = y;
    r[2] = z;
}

inline Vec3D::Vec3D(Vec3D const& source)
{
    r[0] = source.r[0];
    r[1] = source.r[1];
    r[2] = source.r[2];
}

inline Vec3D& Vec3D::operator=(Vec3D const& rhs)
{
    r[0] = rhs.r[0];
    r[1] = rhs.r[1];
    r[2] = rhs.r[2];

    return *this;
}

inline Vec3D& Vec3D::operator+=(Vec3D const& rhs)
{
    r[0] += rhs.r[0];
    r[1] += rhs.r[1];
    r[2] += rhs.r[2];

    return *this;
}

inline Vec3D& Vec3D::operator-=(Vec3D const& rhs)
{
    r[0] -= rhs.r[0];
    r[1] -= rhs.r[1];
    r[2] -= rhs.r[2];

    return *this;
}

inline Vec3D& Vec3D::operator*=(double const a)
{
    r[0] *= a;
    r[1] *= a;
    r[2] *= a;

    return *this;
}

inline Vec3D& Vec3D::operator/=(double const a)
{
    *this *= 1.0 / a;

    return *this;
}

inline double Vec3D::operator*(Vec3D const& v) const
{
    return r[0] * v.r[0] + r[1] * v.r[1] + r[2] * v.r[2];
}

// Doxygen requires namespace prefix for arguments...
inline double& Vec3D::operator[](std::size_t const index)
{
    if (index < 3) return r[index];
    else
    {
        throw std::runtime_error("ERROR: 3D vector has only three"
                                 " components.\n");
    }
}

// Doxygen requires namespace prefix for arguments...
inline double const& Vec3D::operator[](std::size_t const index) const
{
    if (index < 3) return r[index];
    else
    {
        throw std::runtime_error("ERROR: 3D vector has only three"
                                 " components.\n");
    }
}

inline bool Vec3D::operator==(Vec3D const& rhs) const
{
    if (r[0] != rhs.r[0]) return false;
    if (r[1] != rhs.r[1]) return false;
    if (r[2] != rhs.r[2]) return false;
    return true;
}

inline bool Vec3D::operator!=(Vec3D const& rhs) const
{
    return !(*this == rhs);
}

inline double Vec3D::norm() const
{
    return sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
}

inline double Vec3D::norm2() const
{
    return r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
}

inline double Vec3D::l1norm() const
{
    return fabs(r[0]) + fabs(r[1]) + fabs(r[2]);
}

inline Vec3D& Vec3D::normalize()
{
    double n = norm();
    r[0] /= n;
    r[1] /= n;
    r[2] /= n;

    return *this;
}

inline Vec3D Vec3D::cross(Vec3D const& v) const
{
    Vec3D w;

    w.r[0] = r[1] * v.r[2] - r[2] * v.r[1];
    w.r[1] = r[2] * v.r[0] - r[0] * v.r[2];
    w.r[2] = r[0] * v.r[1] - r[1] * v.r[0];

    return w;
}

inline Vec3D operator+(Vec3D lhs, Vec3D const& rhs)
{
    return lhs += rhs;
}

inline Vec3D operator-(Vec3D lhs, Vec3D const& rhs)
{
    return lhs -= rhs;
}

inline Vec3D operator-(Vec3D v)
{
    v *= -1.0;
    return v;
}

inline Vec3D operator*(Vec3D v, double const a)
{
    return v *= a;
}

inline Vec3D operator/(Vec3D v, double const a)
{
    return v /= a;
}

inline Vec3D operator*(double const a, Vec3D v)
{
    return v *= a;
}

}

#endif
