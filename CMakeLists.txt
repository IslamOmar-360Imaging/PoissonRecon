cmake_minimum_required( VERSION 3.10 )

project(poisson_recon)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# external dependencies
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CPM)
# NOTE: to have stable version would should fork this repo.
CPMAddPackage(
	NAME             PoissonRecon
	GIT_REPOSITORY   git@github.com:mkazhdan/PoissonRecon.git
	GIT_TAG          f1c71fe491828d00d34f95de9c264d4bf3481d61
	DOWNLOAD_ONLY
)

add_library( ${PROJECT_NAME} STATIC )

# To remove very annoying "NO ARGUMENTS THAT DEPEND ON A TEMPLATE PARAMETER" errors with G++!
if( UNIX OR MINGW )
	target_compile_options( ${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fpermissive -w> )
elseif( MSVC )
	target_compile_options( ${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj> )
endif()

target_sources( ${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/src/PoissonReconLib.h
		${CMAKE_CURRENT_SOURCE_DIR}/src/PointData.h
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src/PoissonReconLib.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/PointData.cpp
)

target_include_directories( ${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/src
	PRIVATE
		${PoissonRecon_SOURCE_DIR}/Src
)

target_compile_definitions( ${PROJECT_NAME}
	PRIVATE
		_CRT_SECURE_NO_DEPRECATE
		_CRT_SECURE_NO_WARNINGS
		NOMINMAX
)

option( POISSON_RECON_WITH_OPEN_MP "Compile PoissonRecon plugin with OpenMP support" ON )
if( POISSON_RECON_WITH_OPEN_MP )
	find_package( OpenMP )
	
	if( OpenMP_CXX_FOUND ) 
		target_link_libraries( ${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX )
		target_compile_definitions( ${PROJECT_NAME} PRIVATE WITH_OPENMP )
	endif()
endif ()
