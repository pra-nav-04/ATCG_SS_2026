#include "opg/scene/properties.h"

namespace opg {


Properties::Properties()
{}

Properties::~Properties()
{}

bool Properties::hasPropertyWithName(const std::string &name)
{
    auto it = m_values.find(name);
    if (it == m_values.end())
        return false;
    return true;
}

template <typename T>
bool Properties::hasProperty(const std::string &name) const
{
    auto it = m_values.find(name);
    if (it == m_values.end() || !std::holds_alternative<T>(it->second))
        return false;
    return true;
}

template <typename T>
Properties &Properties::setProperty(const std::string &name, T value)
{
    // TODO check if property already exists?
    m_values[name] = value;
    return *this;
}

template <typename T>
std::optional<T> Properties::getProperty(const std::string &name) const
{
    auto it = m_values.find(name);
    if (it == m_values.end() || !std::holds_alternative<T>(it->second))
        return {}; // empty optional<T>
    return std::get<T>(it->second);
}

bool Properties::hasComponent(const std::string &name) const
{
    return hasProperty<SceneComponent *>(name);
}

Properties &Properties::setComponent(const std::string &name, SceneComponent *value)
{
    setProperty<SceneComponent *>(name, value);
    return *this;
}

SceneComponent *Properties::getComponent(const std::string &name) const
{
    return getProperty<SceneComponent *>(name).value_or(nullptr);
}

bool Properties::hasBool(const std::string &name) const
{
    return hasProperty<bool>(name);
}

Properties &Properties::setBool(const std::string &name, bool value)
{
    setProperty<bool>(name, value);
    return *this;
}

bool Properties::getBool(const std::string &name, bool default_value) const
{
    return getProperty<bool>(name).value_or(default_value);
}

std::optional<bool> Properties::getBool(const std::string &name) const
{
    return getProperty<bool>(name);
}

bool Properties::hasFloat(const std::string &name) const
{
    return hasProperty<float>(name);
}

Properties &Properties::setFloat(const std::string &name, float value)
{
    setProperty<float>(name, value);
    return *this;
}

float Properties::getFloat(const std::string &name, float default_value) const
{
    return getProperty<float>(name).value_or(default_value);
}

std::optional<float> Properties::getFloat(const std::string &name) const
{
    return getProperty<float>(name);
}

bool Properties::hasInt(const std::string &name) const
{
    return hasProperty<int>(name);
}

Properties &Properties::setInt(const std::string &name, int value)
{
    setProperty<int>(name, value);
    return *this;
}

int Properties::getInt(const std::string &name, int default_value) const
{
    return getProperty<int>(name).value_or(default_value);
}

std::optional<int> Properties::getInt(const std::string &name) const
{
    return getProperty<int>(name);
}

bool Properties::hasVector(const std::string &name) const
{
    return hasProperty<glm::vec4>(name);
}

Properties &Properties::setVector(const std::string &name, const glm::vec4 &value)
{
    setProperty<glm::vec4>(name, value);
    return *this;
}

glm::vec4 Properties::getVector(const std::string &name, const glm::vec4 &default_value) const
{
    return getProperty<glm::vec4>(name).value_or(default_value);
}

std::optional<glm::vec4> Properties::getVector(const std::string &name) const
{
    return getProperty<glm::vec4>(name);
}

bool Properties::hasMatrix(const std::string &name) const
{
    return hasProperty<glm::mat4>(name);
}

Properties &Properties::setMatrix(const std::string &name, const glm::mat4 &value)
{
    setProperty<glm::mat4>(name, value);
    return *this;
}

glm::mat4 Properties::getMatrix(const std::string &name, const glm::mat4 &default_value) const
{
    return getProperty<glm::mat4>(name).value_or(default_value);
}

std::optional<glm::mat4> Properties::getMatrix(const std::string &name) const
{
    return getProperty<glm::mat4>(name);
}

bool Properties::hasString(const std::string &name) const
{
    return hasProperty<std::string>(name);
}

Properties &Properties::setString(const std::string &name, const std::string &value)
{
    setProperty<std::string>(name, value);
    return *this;
}

std::string Properties::getString(const std::string &name, const std::string &default_value) const
{
    return getProperty<std::string>(name).value_or(default_value);
}

std::optional<std::string> Properties::getString(const std::string &name) const
{
    return getProperty<std::string>(name);
}

bool Properties::hasFloatData(const std::string &name) const
{
    return hasProperty<std::vector<float>>(name);
}

Properties &Properties::setFloatData(const std::string &name, std::vector<float> value)
{
    setProperty<std::vector<float>>(name, std::move(value));
    return *this;
}

std::vector<float> Properties::getFloatData(const std::string &name, std::vector<float> default_value) const
{
    return getProperty<std::vector<float>>(name).value_or(std::move(default_value));
}

std::optional<std::vector<float>> Properties::getFloatData(const std::string &name) const
{
    return getProperty<std::vector<float>>(name);
}

bool Properties::hasRawData(const std::string &name) const
{
    return hasProperty<std::vector<char>>(name);
}

Properties &Properties::setRawData(const std::string &name, std::vector<char> value)
{
    setProperty<std::vector<char>>(name, std::move(value));
    return *this;
}

std::vector<char> Properties::getRawData(const std::string &name, std::vector<char> default_value) const
{
    return getProperty<std::vector<char>>(name).value_or(std::move(default_value));
}

std::optional<std::vector<char>> Properties::getRawData(const std::string &name) const
{
    return getProperty<std::vector<char>>(name);
}

void Properties::copyFrom(const Properties &other, const std::string &src_name, const std::string &dst_name)
{
    m_values[dst_name] = other.m_values.at(src_name);
}

} // end namespace opg
