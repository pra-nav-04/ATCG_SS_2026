#pragma once

#include <variant>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "opg/opgapi.h"
#include "opg/glmwrapper.h"

namespace opg {

class SceneComponent;

class Properties
{
public:
    OPG_API Properties();
    OPG_API ~Properties();

    inline Properties &setPath(const std::string &path) { m_path = path; return *this; }
    inline const std::string &getPath() const { return m_path; }

    OPG_API bool hasComponent(const std::string &name) const;
    OPG_API Properties &setComponent(const std::string &name, SceneComponent *value);
    OPG_API SceneComponent *getComponent(const std::string &name) const;
    template <typename T>
    inline T * getComponentAs(const std::string &name) const { return dynamic_cast<T*>(getComponent(name)); }

    OPG_API bool hasBool(const std::string &name) const;
    OPG_API Properties &setBool(const std::string &name, bool value);
    OPG_API bool getBool(const std::string &name, bool default_value) const;
    OPG_API std::optional<bool> getBool(const std::string &name) const;

    OPG_API bool hasFloat(const std::string &name) const;
    OPG_API Properties &setFloat(const std::string &name, float value);
    OPG_API float getFloat(const std::string &name, float default_value) const;
    OPG_API std::optional<float> getFloat(const std::string &name) const;

    OPG_API bool hasInt(const std::string &name) const;
    OPG_API Properties &setInt(const std::string &name, int value);
    OPG_API int getInt(const std::string &name, int default_value) const;
    OPG_API std::optional<int> getInt(const std::string &name) const;

    OPG_API bool hasVector(const std::string &name) const;
    OPG_API Properties &setVector(const std::string &name, const glm::vec4 &value);
    OPG_API glm::vec4 getVector(const std::string &name, const glm::vec4 &default_value) const;
    OPG_API std::optional<glm::vec4> getVector(const std::string &name) const;

    OPG_API bool hasMatrix(const std::string &name) const;
    OPG_API Properties &setMatrix(const std::string &name, const glm::mat4 &value);
    OPG_API glm::mat4 getMatrix(const std::string &name, const glm::mat4 &default_value) const;
    OPG_API std::optional<glm::mat4> getMatrix(const std::string &name) const;

    OPG_API bool hasString(const std::string &name) const;
    OPG_API Properties &setString(const std::string &name, const std::string &value);
    OPG_API std::string getString(const std::string &name, const std::string &default_value) const;
    OPG_API std::optional<std::string> getString(const std::string &name) const;

    OPG_API bool hasFloatData(const std::string &name) const;
    OPG_API Properties &setFloatData(const std::string &name, std::vector<float> value);
    OPG_API std::vector<float> getFloatData(const std::string &name, std::vector<float> default_value) const;
    OPG_API std::optional<std::vector<float>> getFloatData(const std::string &name) const;

    OPG_API bool hasRawData(const std::string &name) const;
    OPG_API Properties &setRawData(const std::string &name, std::vector<char> value);
    OPG_API std::vector<char> getRawData(const std::string &name, std::vector<char> default_value) const;
    OPG_API std::optional<std::vector<char>> getRawData(const std::string &name) const;

    OPG_API void copyFrom(const Properties &other, const std::string &src_name, const std::string &dst_name);
    OPG_API bool hasPropertyWithName(const std::string &name);

private:

    template <typename T>
    bool hasProperty(const std::string &name) const;
    template <typename T>
    Properties &setProperty(const std::string &name, T value);
    template <typename T>
    std::optional<T> getProperty(const std::string &name) const;

    typedef std::variant<bool, int, float, glm::vec4, glm::mat4, std::vector<float>, std::vector<char>, std::string, SceneComponent*> Value;
    std::unordered_map<std::string, Value> m_values;

    // Path to "root" directory of scene file
    std::string m_path;
};

} // end namespace opg
