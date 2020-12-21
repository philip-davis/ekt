#include <stdio.h>

#include <gtest/gtest.h>

TEST(ekt_simple, simple1)
{
    printf("test\n");

    int a = 1;

    ASSERT_EQ(a, 1);
}

int main(int argc, char **argv)
{

    int result;
    ::testing::InitGoogleTest(&argc, argv);
    result = RUN_ALL_TESTS();

    return result;
}
