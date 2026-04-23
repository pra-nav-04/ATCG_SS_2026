#include "opg/raytracing/shaderbindingtable.h"
#include "opg/exception.h"
#include "opg/hostdevice/misc.h"

#include "opg/raytracing/opg_optix_stubs.h"

#include <algorithm>

namespace opg {

ShaderBindingTable::ShaderBindingTable(OptixDeviceContext context) :
    m_context { context }
{
}

ShaderBindingTable::~ShaderBindingTable()
{
}

uint32_t ShaderBindingTable::addRaygenEntry(OptixProgramGroup prog_group, std::vector<std::byte> sbt_record_custom_data)
{
    uint32_t raygen_sbt_index = static_cast<uint32_t>(m_sbt_entries_raygen.size());
    m_max_raygen_data_size = std::max(m_max_raygen_data_size, sbt_record_custom_data.size());
    m_sbt_entries_raygen.push_back({prog_group, std::move(sbt_record_custom_data)});
    return raygen_sbt_index;
}

uint32_t ShaderBindingTable::addMissEntry(OptixProgramGroup prog_group, std::vector<std::byte> sbt_record_custom_data)
{
    uint32_t miss_sbt_index = static_cast<uint32_t>(m_sbt_entries_miss.size());
    m_max_miss_data_size = std::max(m_max_miss_data_size, sbt_record_custom_data.size());
    m_sbt_entries_miss.push_back({prog_group, std::move(sbt_record_custom_data)});
    return miss_sbt_index;
}

uint32_t ShaderBindingTable::addHitEntry(OptixProgramGroup prog_group, std::vector<std::byte> sbt_record_custom_data)
{
    uint32_t hitgroup_sbt_index = static_cast<uint32_t>(m_sbt_entries_hitgroup.size());
    m_max_hitgroup_data_size = std::max(m_max_hitgroup_data_size, sbt_record_custom_data.size());
    m_sbt_entries_hitgroup.push_back({prog_group, std::move(sbt_record_custom_data)});
    return hitgroup_sbt_index;
}

uint32_t ShaderBindingTable::addCallableEntry(OptixProgramGroup prog_group, std::vector<std::byte> sbt_record_custom_data)
{
    uint32_t callable_sbt_index = static_cast<uint32_t>(m_sbt_entries_callable.size());
    m_max_callable_data_size = std::max(m_max_callable_data_size, sbt_record_custom_data.size());
    m_sbt_entries_callable.push_back({prog_group, std::move(sbt_record_custom_data)});
    return callable_sbt_index;
}


void ShaderBindingTable::createSBT()
{
    std::vector<std::byte> records_data;

    auto add_records = [&records_data](size_t record_stride, auto &sbt_entries)
    {
        // All entries of the same kind must occupy the same amount of storage
        std::vector<std::byte> temp_record_data(record_stride);
        // Pointer to the header in temporary data
        std::byte *temp_record_header_ptr = temp_record_data.data();
        // Pointer to the data segment in temporary data
        std::byte *temp_record_data_ptr = temp_record_data.data() + OPTIX_SBT_RECORD_HEADER_SIZE;
        for (const auto [ prog_group, data ] : sbt_entries)
        {
            // Write record header
            OPTIX_CHECK( optixSbtRecordPackHeader(prog_group, temp_record_header_ptr) );
            // Write record data
            std::copy(data.begin(), data.end(), temp_record_data_ptr);

            // Append new record to records_data
            records_data.insert(records_data.end(), temp_record_data.begin(), temp_record_data.end());
        }
    };

    // TODO align the strides!!!

    // Compute the stride of each entry type in the SBT
    uint32_t raygen_stride   = static_cast<uint32_t>(ceil_to_multiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE + m_max_raygen_data_size,   OPTIX_SBT_RECORD_ALIGNMENT));
    uint32_t miss_stride     = static_cast<uint32_t>(ceil_to_multiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE + m_max_miss_data_size,     OPTIX_SBT_RECORD_ALIGNMENT));
    uint32_t hitgroup_stride = static_cast<uint32_t>(ceil_to_multiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE + m_max_hitgroup_data_size, OPTIX_SBT_RECORD_ALIGNMENT));
    uint32_t callable_stride = static_cast<uint32_t>(ceil_to_multiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE + m_max_callable_data_size, OPTIX_SBT_RECORD_ALIGNMENT));

    // Compute the offset for the start of each entry type in the SBT based on the preceeding entries in our records data
    uint32_t raygen_offset   = 0ul;
    uint32_t miss_offset     = raygen_offset   + raygen_stride   * static_cast<uint32_t>(m_sbt_entries_raygen.size());
    uint32_t hitgroup_offset = miss_offset     + miss_stride     * static_cast<uint32_t>(m_sbt_entries_miss.size());
    uint32_t callable_offset = hitgroup_offset + hitgroup_stride * static_cast<uint32_t>(m_sbt_entries_hitgroup.size());
    uint32_t sbt_size        = callable_offset + callable_stride * static_cast<uint32_t>(m_sbt_entries_callable.size());

    // Reserve enough memory to avoid intermediate allocations
    records_data.reserve(sbt_size);

    // Fill records_data with content!
    add_records(raygen_stride,   m_sbt_entries_raygen);
    add_records(miss_stride,     m_sbt_entries_miss);
    add_records(hitgroup_stride, m_sbt_entries_hitgroup);
    add_records(callable_stride, m_sbt_entries_callable);

    if (records_data.size() != sbt_size)
        throw std::runtime_error("Error encountered while computing SBT data!");

    // Upload records_data to GPU
    m_sbt_buffer.alloc(records_data.size());
    m_sbt_buffer.upload(records_data.data());

    // Allocate a separate shader binding table structure for every raygen program.
    m_sbt.resize(m_sbt_entries_raygen.size());

    // For all raygen programs...
    for (uint32_t raygen_index = 0; raygen_index < m_sbt.size(); ++raygen_index)
    {
        // Alias for the sbt we are currently filling...
        auto& sbt = m_sbt[raygen_index];

        sbt.raygenRecord                = m_sbt_buffer.getRaw(raygen_offset + raygen_index * raygen_stride);

        if (!m_sbt_entries_miss.empty())
        {
            sbt.missRecordBase              = m_sbt_buffer.getRaw(miss_offset);
            sbt.missRecordStrideInBytes     = miss_stride;
            sbt.missRecordCount             = static_cast<uint32_t>(m_sbt_entries_miss.size());
        }
        else
        {
            sbt.missRecordBase              = 0;
            sbt.missRecordStrideInBytes     = 0;
            sbt.missRecordCount             = 0;
        }

        if (!m_sbt_entries_hitgroup.empty())
        {
            sbt.hitgroupRecordBase          = m_sbt_buffer.getRaw(hitgroup_offset);
            sbt.hitgroupRecordStrideInBytes = hitgroup_stride;
            sbt.hitgroupRecordCount         = static_cast<uint32_t>(m_sbt_entries_hitgroup.size());
        }
        else
        {
            sbt.hitgroupRecordBase          = 0;
            sbt.hitgroupRecordStrideInBytes = 0;
            sbt.hitgroupRecordCount         = 0;
        }

        if (!m_sbt_entries_callable.empty())
        {
            sbt.callablesRecordBase          = m_sbt_buffer.getRaw(callable_offset);
            sbt.callablesRecordStrideInBytes = callable_stride;
            sbt.callablesRecordCount         = static_cast<uint32_t>(m_sbt_entries_callable.size());
        }
        else
        {
            sbt.callablesRecordBase          = 0;
            sbt.callablesRecordStrideInBytes = 0;
            sbt.callablesRecordCount         = 0;
        }
    }
}

} // end namespace opg
