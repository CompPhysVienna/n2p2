###################################################################################################
# CMake Build System for n2p2 executables
###################################################################################################

###################################################################################################
# Specify executables based on dependencies
# APP_CORE require only libnnp
set(APP_CORE
        nnp-convert
        nnp-cutoff
        nnp-dist
        nnp-predict
        nnp-prune
        nnp-select
        nnp-symfunc)

# APP_DATASET require libnnp and libnnptrain (Use class Dataset only, not Training)
# NB: These are separated out since they do not require the Eigen library
set(APP_DATASET
        nnp-atomenv
        nnp-checkf
        nnp-comp2
        nnp-dataset
        nnp-norm
        nnp-scaling)

# APP_TRAINING require libnnp and libnnptrain
set(APP_TRAINING
        nnp-checkdw
        nnp-norm2
        nnp-train)

# APP contains all executables
set(APP ${APP_CORE} ${APP_DATASET} ${APP_TRAINING})
###################################################################################################

###################################################################################################
# Specify Build Target for Each APP
foreach(app IN ITEMS ${APP})
    add_executable(${app} ${app}.cpp)
endforeach(app IN ITEMS ${APP})

# Set Directory-Wide Project Options
add_compile_options(${PROJECT_OPTIONS})
###################################################################################################

###################################################################################################
# Find Required Package dependencies
include_directories(${MPI_INCLUDE_PATH} ${GSL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/libnnp ${PROJECT_SOURCE_DIR}/libnnptrain)

# Add Link Dependencies based on requirements
foreach(app IN ITEMS ${APP_CORE})
    target_link_libraries(${app} PRIVATE nnp)
endforeach(app IN ITEMS ${APP_CORE})

foreach(app IN ITEMS ${APP_DATASET})
    target_link_libraries(${app} PRIVATE ${GSL_LIBRARIES}  nnp nnptrain)
endforeach(app IN ITEMS ${APP_DATASET})

foreach(app IN ITEMS ${APP_TRAINING})
    target_link_libraries(${app} PRIVATE ${GSL_LIBRARIES}  Eigen3::Eigen nnp nnptrain)
endforeach(app IN ITEMS ${APP_TRAINING})

###################################################################################################]]
