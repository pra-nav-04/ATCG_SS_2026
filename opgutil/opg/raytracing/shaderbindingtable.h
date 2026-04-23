#pragma once

#include <vector>
#include <memory>

#include <optix.h>

#include "opg/memory/devicebuffer.h"
#include "opg/opgapi.h"

namespace opg {

struct HitGroupSBTData
{
    void *shape_data_ptr;
    void *bsdf_data_ptr;
};

class ShaderBindingTable
{
public:
    OPG_API ShaderBindingTable(OptixDeviceContext context);
    OPG_API ~ShaderBindingTable();

    // Add an SBT entry without any additional data stored in the SBT, return the index within the respective entry class
    inline uint32_t addRaygenEntry  (OptixProgramGroup prog_group) { return addRaygenEntry(prog_group, std::vector<std::byte>()); }
    inline uint32_t addMissEntry    (OptixProgramGroup prog_group) { return addMissEntry(prog_group, std::vector<std::byte>()); }
    inline uint32_t addHitEntry     (OptixProgramGroup prog_group) { return addHitEntry(prog_group, std::vector<std::byte>()); }
    inline uint32_t addCallableEntry(OptixProgramGroup prog_group) { return addCallableEntry(prog_group, std::vector<std::byte>()); }

    // Add an SBT entry where the contents of sbt_record_data is copied into the data segment of the SBT entry, return the index within the respective entry class
    template <typename T>
    inline uint32_t addRaygenEntry  (OptixProgramGroup prog_group, const T &sbt_record_data) { return addRaygenEntry(prog_group, std::vector<std::byte>(reinterpret_cast<const std::byte*>(&sbt_record_data), reinterpret_cast<const std::byte*>(&sbt_record_data + 1))); }
    template <typename T>
    inline uint32_t addMissEntry    (OptixProgramGroup prog_group, const T &sbt_record_data) { return addMissEntry(prog_group, std::vector<std::byte>(reinterpret_cast<const std::byte*>(&sbt_record_data), reinterpret_cast<const std::byte*>(&sbt_record_data + 1))); }
    template <typename T>
    inline uint32_t addHitEntry     (OptixProgramGroup prog_group, const T &sbt_record_data) { return addHitEntry(prog_group, std::vector<std::byte>(reinterpret_cast<const std::byte*>(&sbt_record_data), reinterpret_cast<const std::byte*>(&sbt_record_data + 1))); }
    template <typename T>
    inline uint32_t addCallableEntry(OptixProgramGroup prog_group, const T &sbt_record_data) { return addCallableEntry(prog_group, std::vector<std::byte>(reinterpret_cast<const std::byte*>(&sbt_record_data), reinterpret_cast<const std::byte*>(&sbt_record_data + 1))); }

    // Add an SBT entry with custom data stored in an std::vector, return the index within the respective entry class
    OPG_API uint32_t addRaygenEntry  (OptixProgramGroup prog_group, std::vector<std::byte> sbt_record_custom_data);
    OPG_API uint32_t addMissEntry    (OptixProgramGroup prog_group, std::vector<std::byte> sbt_record_custom_data);
    OPG_API uint32_t addHitEntry     (OptixProgramGroup prog_group, std::vector<std::byte> sbt_record_custom_data);
    OPG_API uint32_t addCallableEntry(OptixProgramGroup prog_group, std::vector<std::byte> sbt_record_custom_data);

    OPG_API void createSBT();

    inline const OptixShaderBindingTable *getSBT(uint32_t raygen_index = 0) const { return raygen_index < m_sbt.size() ? &m_sbt[raygen_index] : nullptr; }

private:
    OptixDeviceContext  m_context;

    size_t m_max_raygen_data_size     = 0;
    size_t m_max_miss_data_size       = 0;
    size_t m_max_hitgroup_data_size   = 0;
    size_t m_max_callable_data_size   = 0;

    std::vector<std::pair<OptixProgramGroup, std::vector<std::byte>>> m_sbt_entries_raygen;
    std::vector<std::pair<OptixProgramGroup, std::vector<std::byte>>> m_sbt_entries_miss;
    std::vector<std::pair<OptixProgramGroup, std::vector<std::byte>>> m_sbt_entries_hitgroup;
    std::vector<std::pair<OptixProgramGroup, std::vector<std::byte>>> m_sbt_entries_callable;

    // typedef SBTRecord<void*> RecordType;

    GenericDeviceBuffer         m_sbt_buffer;

    std::vector<OptixShaderBindingTable> m_sbt;
};

} // end namespace opg








