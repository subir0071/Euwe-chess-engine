#include "FrontEndOption.h"

#include "MyAssert.h"
#include "RangePatches.h"

#include <charconv>
#include <format>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace {

bool stringViewToBool(std::string_view valueString) {
    std::istringstream sstream{std::string(valueString)};
    bool value{};
    sstream >> std::boolalpha >> value;

    if (!sstream) {
        throw std::invalid_argument(std::format("Invalid boolean value: '{}'", valueString));
    }

    return value;
}

int stringViewToInt(std::string_view valueString) {
    int value{};
    const auto result =
            std::from_chars(valueString.data(), valueString.data() + valueString.size(), value);

    if (result.ec != std::errc{}) {
        switch (result.ec) {
            case std::errc::invalid_argument:
                throw std::invalid_argument(
                        std::format("Invalid integer value: '{}'", valueString));
            case std::errc::result_out_of_range:
                throw std::out_of_range(
                        std::format("Integer value out of range: '{}'", valueString));
            default: {
                const auto error_code = std::make_error_code(result.ec);
                throw std::system_error(
                        error_code,
                        std::format(
                                "Unknown error while parsing integer value '{}': error code {}: "
                                "{} ",
                                valueString,
                                error_code.value(),
                                error_code.message()));
            }
        }
    }

    return value;
}

}  // namespace

FrontEndOption FrontEndOption::createAction(std::string name, std::function<void()> onTrigger) {
    FrontEndOption option;
    option.name_  = std::move(name);
    option.type_  = Type::Action;
    option.onSet_ = [onTrigger = std::move(onTrigger)](std::string_view) {
        onTrigger();
    };
    return option;
}

FrontEndOption FrontEndOption::createBoolean(
        std::string name, const bool defaultValue, std::function<void(bool)> onSet) {
    std::ostringstream sstream;
    sstream << std::boolalpha << defaultValue;

    FrontEndOption option;
    option.name_         = std::move(name);
    option.type_         = Type::Boolean;
    option.defaultValue_ = sstream.str();
    option.onSet_        = [onSet = std::move(onSet)](std::string_view valueString) {
        onSet(stringViewToBool(valueString));
    };
    return option;
}

FrontEndOption FrontEndOption::createBoolean(std::string name, bool& value) {
    return createBoolean(std::move(name), value, [&](bool v) { value = v; });
}

FrontEndOption FrontEndOption::createString(
        std::string name, std::string defaultValue, OnSet onSet) {
    FrontEndOption option;
    option.name_         = std::move(name);
    option.type_         = Type::String;
    option.defaultValue_ = std::move(defaultValue);
    option.onSet_        = std::move(onSet);
    return option;
}

FrontEndOption FrontEndOption::createString(std::string name, std::string& value) {
    return createString(std::move(name), value, [&](std::string_view v) { value = v; });
}

FrontEndOption FrontEndOption::createInteger(
        std::string name,
        const int defaultValue,
        const int minValue,
        const int maxValue,
        std::function<void(int)> onSet) {
    FrontEndOption option;
    option.name_         = std::move(name);
    option.type_         = Type::Integer;
    option.defaultValue_ = std::to_string(defaultValue);
    option.minValue_     = minValue;
    option.maxValue_     = maxValue;
    option.onSet_        = [=, onSet = std::move(onSet)](std::string_view valueString) {
        const int value = stringViewToInt(valueString);

        if (value < minValue || value > maxValue) {
            throw std::invalid_argument(std::format(
                    "Value out of range: expected [{}, {}], got {}", minValue, maxValue, value));
        }

        onSet(value);
    };
    return option;
}

FrontEndOption FrontEndOption::createInteger(
        std::string name, int& value, const int minValue, const int maxValue) {
    return createInteger(std::move(name), value, minValue, maxValue, [&](int v) { value = v; });
}

FrontEndOption FrontEndOption::createAlternative(
        std::string name,
        std::string defaultValue,
        std::vector<std::string> validValues,
        OnSet onSet) {
    FrontEndOption option;
    option.name_         = std::move(name);
    option.type_         = Type::Alternative;
    option.validValues_  = std::move(validValues);
    option.defaultValue_ = std::move(defaultValue);
    option.onSet_        = [onSet       = std::move(onSet),
                     validValues = *option.validValues_](std::string_view valueString) {
        const auto it = std::find(validValues.begin(), validValues.end(), valueString);
        if (it == validValues.end()) {
            const std::string validValuesString = validValues | joinToString(", ");
            throw std::invalid_argument(std::format(
                    "Invalid value '{}'. Expected one of: [{}]", valueString, validValuesString));
        }
        onSet(valueString);
    };
    return option;
}

FrontEndOption FrontEndOption::createAlternative(
        std::string name, std::string& value, std::vector<std::string> validValues) {
    return createAlternative(
            std::move(name), value, std::move(validValues), [&](std::string_view v) { value = v; });
}

const std::string& FrontEndOption::retrieveDefaultValue() const {
    MY_ASSERT(type_ != Type::Action);
    MY_ASSERT(defaultValue_.has_value());
    return *defaultValue_;
}

int FrontEndOption::retrieveMinValue() const {
    MY_ASSERT(type_ == Type::Integer);
    MY_ASSERT(minValue_.has_value());
    return *minValue_;
}

int FrontEndOption::retrieveMaxValue() const {
    MY_ASSERT(type_ == Type::Integer);
    MY_ASSERT(maxValue_.has_value());
    return *maxValue_;
}

const std::vector<std::string>& FrontEndOption::retrieveValidValues() const {
    MY_ASSERT(type_ == Type::Alternative);
    MY_ASSERT(validValues_.has_value());
    return *validValues_;
}

void FrontEndOption::set(std::string_view valueString) {
    onSet_(valueString);
}

void FrontEndOption::trigger() {
    if (type_ != Type::Action) {
        throw std::logic_error("Cannot set value for action option");
    }

    onSet_("");
}
