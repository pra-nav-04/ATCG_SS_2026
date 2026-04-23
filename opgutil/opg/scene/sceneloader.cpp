#include "opg/scene/sceneloader.h"

#include "opg/opg.h"

#include <sstream>
#include <algorithm>
#include <pugixml.hpp>
#include <fmt/core.h>

namespace opg {

class SceneComponentRegistry
{
public:
    void registerFactory(const std::string &name, SceneComponentFactory factory)
    {
        auto &known_factory = m_known_factories[name];
        OPG_ASSERT_MSG(!known_factory, fmt::format("Factory with name '{}' already registered!", name));
        known_factory = factory;
    }

    SceneComponentFactory getFactory(const std::string &name)
    {
        auto factory = m_known_factories[name];
        OPG_ASSERT_MSG(factory, fmt::format("Factory with name '{}' does not exist!", name));
        return factory;
    }

private:
    std::unordered_map<std::string, SceneComponentFactory> m_known_factories;

// Singleton access:
public:
    static SceneComponentRegistry *getInstance() { if (!m_instance) m_instance = std::make_unique<SceneComponentRegistry>(); return m_instance.get(); }
private:
    static std::unique_ptr<SceneComponentRegistry> m_instance;
};

std::unique_ptr<SceneComponentRegistry> SceneComponentRegistry::m_instance;

void registerSceneComponentFactory(const std::string &name, SceneComponentFactory factory)
{
    auto registry = SceneComponentRegistry::getInstance();
    registry->registerFactory(name, std::move(factory));
}


XMLSceneLoader::XMLSceneLoader(Scene *scene) :
    m_scene { scene },
    m_registry { SceneComponentRegistry::getInstance() }
{
}

XMLSceneLoader::~XMLSceneLoader()
{
}

glm::mat4 XMLSceneLoader::parseTransformFromXML(const pugi::xml_node &transform_node)
{
    glm::mat4 result = glm::mat4(1);

    for (pugi::xml_node child_node : transform_node.children())
    {
        std::string name = child_node.name();
        if (false)
        {
        }
        else if (name == "matrix")
        {
            std::stringstream ss(child_node.text().as_string());
            glm::mat4 matrix;
            for (int r = 0; r < 4; ++r)
            {
                for (int c = 0; c < 4; ++c)
                {
                    ss >> matrix[c][r];
                }
            }
            // There should not have been a failure during loading of the matrix coefficients
            OPG_CHECK(!ss.fail());
            result = result * matrix;
        }
        else if (name == "rotate")
        {
            float angle = child_node.attribute("angle").as_float() * M_PIf/180.0f;
            float x = child_node.attribute("x").as_float();
            float y = child_node.attribute("y").as_float();
            float z = child_node.attribute("z").as_float();
            result = glm::rotate(result, angle, glm::normalize(glm::vec3(x,y,z)));
        }
        else if (name == "scale")
        {
            float x = child_node.attribute("x").as_float();
            float y = child_node.attribute("y").as_float();
            float z = child_node.attribute("z").as_float();
            result = glm::scale(result, glm::vec3(x,y,z));
        }
        else if (name == "translate")
        {
            float x = child_node.attribute("x").as_float();
            float y = child_node.attribute("y").as_float();
            float z = child_node.attribute("z").as_float();
            result = glm::translate(result, glm::vec3(x,y,z));
        }
        else if (name == "lookat")
        {
            auto parse_vec3 = [](const pugi::xml_attribute &attrib)->glm::vec3 {
                std::vector<std::string> tokens = splitString(attrib.as_string(), ", ");
                OPG_CHECK(tokens.size() == 3);
                return glm::vec3(std::atof(tokens[0].c_str()), std::atof(tokens[1].c_str()), std::atof(tokens[2].c_str()));
            };
            glm::vec3 eye = parse_vec3(child_node.attribute("eye"));
            glm::vec3 center = parse_vec3(child_node.attribute("center"));
            glm::vec3 up = parse_vec3(child_node.attribute("up"));
            glm::mat4 lookat = glm::lookAt(eye, center, up);
            result = result * lookat;
        }
    }

    return result;
}

void XMLSceneLoader::parsePropertyFromXML(const pugi::xml_node &prop_node, Properties &props)
{
    std::string name = prop_node.attribute("name").as_string();
    std::string type = prop_node.attribute("type").as_string();

    if (false)
    {
    }
    else if (type == "bool")
    {
        // If the property is present, but without value, we assume this is like a "flag" that is raised.
        props.setBool(name, prop_node.attribute("value").as_bool(true));
    }
    else if (type == "int")
    {
        props.setInt(name, prop_node.attribute("value").as_int());
    }
    else if (type == "float")
    {
        props.setFloat(name, prop_node.attribute("value").as_float());
    }
    else if (type == "vector")
    {
        float x = prop_node.attribute("x").as_float(0);
        float y = prop_node.attribute("y").as_float(0);
        float z = prop_node.attribute("z").as_float(0);
        float w = prop_node.attribute("w").as_float(1);
        props.setVector(name, glm::vec4(x, y, z, w));
    }
    else if (type == "transform")
    {
        glm::mat4 transform = parseTransformFromXML(prop_node);
        props.setMatrix(name, transform);
    }
    else if (type == "string")
    {
        props.setString(name, prop_node.attribute("value").as_string());
    }
    else if (type == "floatdata")
    {
        std::stringstream ss; // (prop_node.text().as_string());
        for (auto child : prop_node.children())
        {
            ss << child.text().as_string();
        }

        std::vector<float> data;
        float f;
        while (ss >> f)
        {
            data.push_back(f);
        }
        OPG_CHECK_MSG(ss.eof(), "Parsing float data did not reach end of stream!");

        props.setFloatData(name, data);
    }
    else if (type == "rawdata")
    {
        std::stringstream ss; // (prop_node.text().as_string());
        for (auto child : prop_node.children())
        {
            ss << child.text().as_string();
        }

        // Configure numeric base of the data
        std::string base = prop_node.attribute("base").as_string("dec");
        if (base == "dec")
            ss >> std::dec;
        else if (base == "hex")
            ss >> std::hex;
        else if (base == "oct")
            ss >> std::oct;
        else
            OPG_CHECK(false);

        int width = prop_node.attribute("width").as_int(1);
        OPG_CHECK(width == 1 || width == 2 || width == 4 || width == 8);

        std::vector<char> data;
        int64_t v;
        while (ss >> v)
        {
            OPG_CHECK_MSG(v >> (8*width) == 0 || v >> (8*width) == ~0, fmt::format("Raw data value '{}' is out of range!", v));
            // Add the lower "width" number of bytes from v to data
            data.insert(data.end(), reinterpret_cast<char*>(&v), reinterpret_cast<char*>(&v) + width);
        }
        OPG_CHECK_MSG(ss.eof(), "Parsing float data did not reach end of stream!");

        props.setRawData(name, data);
    }
    else if (type == "reference")
    {
        std::string ref_name = prop_node.attribute("value").as_string();
        SceneComponent *component = m_scene->getSceneComponentByName(ref_name);
        props.setComponent(name, component);
    }
}

SceneComponent *XMLSceneLoader::createComponentFromXML(const pugi::xml_node &component_node, const std::string &name_prefix)
{
    std::string type = component_node.attribute("type").as_string();
    std::string name = component_node.attribute("name").as_string();

    if (!name_prefix.empty())
        name = name_prefix + "." + name;

    // compute prpos.

    Properties props;
    props.setPath(m_path);
    props.setString("name", name);
    for (const pugi::xml_node child_node : component_node.children())
    {
        std::string node_name = child_node.name();

        if (false)
        {}
        /*
        else if (node_name == "component")
        {
            // Parse sub component, well....
        }
        */
        else if (node_name == "property")
        {
            parsePropertyFromXML(child_node, props);
        }
    }

    auto factory = m_registry->getFactory(type);
    return factory(m_scene, props);
}

void XMLSceneLoader::loadFromFile(const std::string &filename)
{
    std::vector<char> filedata = readFile(filename.c_str());
    pugi::xml_document doc;
    auto result = doc.load_buffer(filedata.data(), filedata.size());
    // auto result = doc.load_file(filename.c_str());
    if (!result)
    {
        size_t line_number = 1;
        size_t line_begin = 0;
        for (size_t i = 0; i < result.offset; ++i)
        {
            if (filedata[i] == '\n')
            {
                line_number++;
                line_begin = i+1;
            }
        }
        size_t line_end = std::string::npos;
        for (size_t i = result.offset; i < filedata.size(); ++i)
        {
            if (filedata[i] == '\n')
            {
                line_end = i;
                break;
            }
        }
        std::string error_line = std::string(filedata.begin() + line_begin, filedata.begin() + line_end);

        throw std::runtime_error(fmt::format("Failed to load xml document: {} at line {}:\n{}", result.description(), line_number, error_line));
        //std::stringstream ss;
        //ss << "Failed to load xml document: " << result.description();
        //ss << " at line " << line_number << ":" << std::endl;
        //ss << error_line;
        //throw std::runtime_error(ss.str());
    }

    // Compute the root directory of the scene file
    m_path.clear();
    auto last_forward_sep_idx = filename.find_last_of('/');
    auto last_backward_sep_idx = filename.find_last_of('\\');
    auto last_sep_idx = last_forward_sep_idx == std::string::npos ? last_backward_sep_idx :
                        last_backward_sep_idx == std::string::npos ? last_forward_sep_idx :
                        std::max(last_forward_sep_idx, last_backward_sep_idx);
    if (last_sep_idx != std::string::npos)
        m_path = filename.substr(0, last_sep_idx+1); // include separator in path!

    // Load all components declared in the xml file
    for (pugi::xml_node component_node : doc.child("scene").children("component"))
    {
        createComponentFromXML(component_node);
    }
}

} // end namespace opg
