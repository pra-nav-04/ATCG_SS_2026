#pragma once

// Forward declarations...
struct EmitterVPtrTable;
struct BSDFVPtrTable;
struct PhaseFunctionVPtrTable;
struct MediumVPtrTable;


// This struct is the base for storing common the information for rays interacting with the scene.
// There are two specializations:
// - SurfaceInteraction: These interactions represent the result of a ray tracing operation where the ray intersects the scene geometry.
// - MediumInteraction: These interactions represent the result of a ray interacting with participating media in "free space". (This will only become relevant lateron)
struct Interaction
{
    // The position at which the interaction takes place, i.e. the end point of the ray that resulted in this interaction.
    glm::vec3 position;

    // The direction through which a ray encountered this interaction.
    glm::vec3 incoming_ray_dir;
    // The distance traveled along the incoming ray direction until the intersection is reached.
    float     incoming_distance;

    // Create an empty Interaction that identifies an invalid (i.e. none) interaction.
    OPG_INLINE OPG_HOSTDEVICE static Interaction empty()
    {
        Interaction result;
        result.position = glm::vec3(std::numeric_limits<float>::signaling_NaN());
        result.incoming_ray_dir = glm::vec3(std::numeric_limits<float>::signaling_NaN());
        result.incoming_distance = std::numeric_limits<float>::signaling_NaN();
        return result;
    }

    // Check if a valid interaction was found.
    // The interaction might still be finite or infinite!
    OPG_INLINE OPG_HOSTDEVICE bool is_valid() const
    {
        // infinite distance counts as valid!
        return !glm::isnan(incoming_distance);
    }

    // Mark the current interaction as invalid.
    OPG_INLINE OPG_HOSTDEVICE void set_invalid()
    {
        incoming_distance = std::numeric_limits<float>::signaling_NaN();
    }

    // Check if the found interaction is finite.
    OPG_INLINE OPG_HOSTDEVICE bool is_finite() const
    {
        return glm::isfinite(incoming_distance);
        //return glm::all(glm::isfinite(position));
    }

    OPG_INLINE OPG_HOSTDEVICE void set_infinite()
    {
        incoming_distance = std::numeric_limits<float>::infinity();
    }
};


// This is a specialization for storing information on a ray-surface interaction.
// In addition to the data in the Interaction struct, we store information on the local surface geometry and its BSDF here.
struct SurfaceInteraction : public Interaction
{
    // Local surface properties:

    // Real geometric normal at the intersection point.
    glm::vec3 geom_normal;

    // Unit vector representing the surface normal at the intersection point.
    glm::vec3 normal;
    // Unit vector representing the tangent vector at the intersection point, which is used for anisotropic effects.
    glm::vec3 tangent;
    // A uv coordinate that could be used to access textures or parameterize any other function on the surface.
    glm::vec2 uv;

    // Index of the primitive hit inside of the shape (Useful for meshes where it indicates which triangle was hit)
    uint32_t primitive_index;


    // Components present at this location:

    // A pointer to the function table of the BSDF at this location, or nullptr if no BSDF is present.
    const BSDFVPtrTable    *bsdf;
    // A pointer to the function table of the emitter at this location, or nullptr if no emitter is present.
    const EmitterVPtrTable *emitter;
    // A pointer to the function table of the medium inside of the geometry (used in volume rendering only).
    const MediumVPtrTable *inside_medium;
    // A pointer to the function table of the medium outside of the geometry (used in volume rendering only).
    const MediumVPtrTable *outside_medium;


    SurfaceInteraction() = default;
    OPG_HOSTDEVICE OPG_INLINE SurfaceInteraction(const Interaction &i) : Interaction(i) {}

    // Create an empty Interaction that identifies an invalid (i.e. none) interaction.
    static OPG_HOSTDEVICE OPG_INLINE SurfaceInteraction empty()
    {
        SurfaceInteraction result = Interaction::empty();
        result.bsdf = nullptr;
        result.emitter = nullptr;
        result.inside_medium = nullptr;
        result.outside_medium = nullptr;
        return result;
    }
};

// This is a specialization for storing information on a ray-volume interaction.
// We don't need to store additional data here (for now).
struct MediumInteraction : public Interaction
{
    MediumInteraction() = default;
    OPG_HOSTDEVICE OPG_INLINE MediumInteraction(const Interaction &i) : Interaction(i) {}

    // Create an empty Interaction that identifies an invalid (i.e. none) interaction.
    static OPG_HOSTDEVICE OPG_INLINE MediumInteraction empty()
    {
        MediumInteraction result = Interaction::empty();
        return result;
    }
};
