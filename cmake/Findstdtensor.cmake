INCLUDE(ExternalProject)
INCLUDE(cmake/deps.cmake)

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

# https://cmake.org/cmake/help/v3.0/module/ExternalProject.html
EXTERNALPROJECT_ADD(stdtensor-repo
                    GIT_REPOSITORY ${STDTENSOR_GIT_URL}
                    GIT_TAG ${STDTENSOR_GIT_TAG}
                    PREFIX ${PREFIX}
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PREFIX}
                               -DBUILD_TESTS=0
                               -DBUILD_EXAMPLES=0
                               -DBUILD_BENCHMARKS=0)

INCLUDE_DIRECTORIES(${PREFIX}/src/stdtensor-repo/include)

FUNCTION(TARGET_USE_STDTENSOR target)
    ADD_DEPENDENCIES(${target} stdtensor-repo)
ENDFUNCTION()
