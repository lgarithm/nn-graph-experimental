INCLUDE(ExternalProject)
INCLUDE(cmake/deps.cmake)

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(cunn-ops-repo
                    GIT_REPOSITORY ${CUNN_OPS_GIT_URL}
                    GIT_TAG ${CUNN_OPS_GIT_TAG}
                    PREFIX ${PREFIX}
                    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
                               -DTTL_HOME=${PREFIX}/src/stdtensor-repo
                               -DNN_OPS_HOME=${PREFIX}/src/stdnn-ops-repo
                               -DCMAKE_INSTALL_PREFIX=${PREFIX})

ADD_DEPENDENCIES(cunn-ops-repo stdtensor-repo)
ADD_DEPENDENCIES(cunn-ops-repo stdnn-ops-repo)
LINK_DIRECTORIES(${PREFIX}/lib)

FUNCTION(TARGET_USE_CUNN_OPS target)
    ADD_DEPENDENCIES(${target} cunn-ops-repo)
    TARGET_LINK_LIBRARIES(${target} cunn-ops)
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${PREFIX}/include)
ENDFUNCTION()
