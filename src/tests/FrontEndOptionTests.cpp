#include "chess-engine-lib/FrontEndOption.h"

#include "MyGTest.h"

#include <stdexcept>

namespace FrontEndOptionTests {

TEST(FrontEndOptionTests, TestAction) {
    bool triggered         = false;
    const std::string name = "action";

    FrontEndOption action = FrontEndOption::createAction(name, [&]() { triggered = true; });

    EXPECT_EQ(action.getName(), name);

    EXPECT_FALSE(action.getDefaultValue().has_value());
    EXPECT_FALSE(action.getMinValue().has_value());
    EXPECT_FALSE(action.getMaxValue().has_value());
    EXPECT_FALSE(action.getValidValues().has_value());

    action.trigger();
    EXPECT_TRUE(triggered);
}

TEST(FrontEndOptionTests, TestBoolean) {
    bool value             = false;
    const std::string name = "boolean";

    FrontEndOption action = FrontEndOption::createBoolean(name, value);

    EXPECT_EQ(action.getName(), name);

    const auto& defaultValue = action.getDefaultValue();
    ENFORCE_TRUE(defaultValue.has_value());
    EXPECT_EQ(defaultValue.value(), "false");

    EXPECT_FALSE(action.getMinValue().has_value());
    EXPECT_FALSE(action.getMaxValue().has_value());
    EXPECT_FALSE(action.getValidValues().has_value());

    EXPECT_THROW(action.trigger(), std::logic_error);

    action.set("true");
    EXPECT_TRUE(value);

    action.set("false");
    EXPECT_FALSE(value);

    EXPECT_THROW(action.set("invalid"), std::invalid_argument);
}

TEST(FrontEndOptionTests, TestString) {
    std::string value      = "default";
    const std::string name = "string";

    FrontEndOption action = FrontEndOption::createString(name, value);

    EXPECT_EQ(action.getName(), name);

    const auto& defaultValue = action.getDefaultValue();
    ENFORCE_TRUE(defaultValue.has_value());
    EXPECT_EQ(defaultValue.value(), value);

    EXPECT_FALSE(action.getMinValue().has_value());
    EXPECT_FALSE(action.getMaxValue().has_value());
    EXPECT_FALSE(action.getValidValues().has_value());

    EXPECT_THROW(action.trigger(), std::logic_error);

    action.set("new value");
    EXPECT_EQ(value, "new value");
}

TEST(FrontEndOptionTests, TestInteger) {
    int value              = 0;
    const std::string name = "integer";

    FrontEndOption action = FrontEndOption::createInteger(name, value, -10, 10);

    EXPECT_EQ(action.getName(), name);

    const auto& defaultValue = action.getDefaultValue();
    ENFORCE_TRUE(defaultValue.has_value());
    EXPECT_EQ(defaultValue.value(), "0");

    const auto& minValue = action.getMinValue();
    ENFORCE_TRUE(minValue.has_value());
    EXPECT_EQ(minValue.value(), -10);

    const auto& maxValue = action.getMaxValue();
    ENFORCE_TRUE(maxValue.has_value());
    EXPECT_EQ(maxValue.value(), 10);

    EXPECT_FALSE(action.getValidValues().has_value());

    EXPECT_THROW(action.trigger(), std::logic_error);

    action.set("5");
    EXPECT_EQ(value, 5);

    EXPECT_THROW(action.set("-11"), std::invalid_argument);
    EXPECT_THROW(action.set("11"), std::invalid_argument);
    EXPECT_EQ(value, 5);
}

TEST(FrontEndOptionTests, TestAlternative) {
    std::string value      = "default";
    const std::string name = "alternative";

    std::vector<std::string> alternatives{"a", "b", "c", "default"};

    FrontEndOption action = FrontEndOption::createAlternative(name, value, alternatives);

    EXPECT_EQ(action.getName(), name);

    const auto& defaultValue = action.getDefaultValue();
    ENFORCE_TRUE(defaultValue.has_value());
    EXPECT_EQ(defaultValue.value(), value);

    EXPECT_FALSE(action.getMinValue().has_value());
    EXPECT_FALSE(action.getMaxValue().has_value());

    const auto& validValues = action.getValidValues();
    ENFORCE_TRUE(validValues.has_value());
    EXPECT_EQ(validValues.value(), alternatives);

    EXPECT_THROW(action.trigger(), std::logic_error);

    action.set("b");
    EXPECT_EQ(value, "b");

    EXPECT_THROW(action.set("invalid"), std::invalid_argument);
    EXPECT_EQ(value, "b");
}

}  // namespace FrontEndOptionTests
