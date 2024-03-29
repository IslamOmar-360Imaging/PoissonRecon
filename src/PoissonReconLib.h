#pragma once

#include <cstddef>

//! Wrapper to use PoissonRecon (Kazhdan et. al) as a library
class PoissonReconLib
{
public:

	//! Algorithm parameters
	struct Parameters
	{
		//! Default initializer
		Parameters();

		//! Boundary types
		enum BoundaryType { FREE, DIRICHLET, NEUMANN };

		//! Boundary type for the finite elements
		BoundaryType boundary = NEUMANN;

		//! The maximum depth of the tree that will be used for surface reconstruction
		/** Running at depth d corresponds to solving on a 2^d x 2^d x 2^d.
			Note that since the reconstructor adapts the octree to the sampling density,
			the specified reconstruction depth is only an upper bound.
		**/
		int depth = 8;

		//! The depth up to which the solver will solve the numerical system. It will still
		//! show the results at the finest resolution, but no additional high-frequency data
		//! will be introduced at the finest resolutions. (It could also be the case that
		//! aliasing that occurs at the coarser resolutions will not get corrected.)
		int solveDepth = 0;

		//! The depth at which sampling density is estimated in order to assign a weight to each sample
		int kernelDepth = 0;

		//! When using Dirichlet envelope constraints, what depth should the envelope surface be rasterized into?
		int envelopeDepth = 0;

		//! The target width of the finest level octree cells (ignored if depth is specified)
		float finestCellWidth = 0.0f;

		//! The ratio between the diameter of the cube used for reconstruction and the diameter of the samples' bounding cube.
		/** Specifies the factor of the bounding cube that the input samples should fit into.
		**/
		float scale = 1.1f;

		//! The minimum number of sample points that should fall within an octree node as the octree construction is adapted to sampling density.
		/** This parameter specifies the minimum number of points that should fall within an octree node.
			For noise-free samples, small values in the range [1.0 - 5.0] can be used. For more noisy samples, larger values
			in the range [15.0 - 20.0] may be needed to provide a smoother, noise-reduced, reconstruction.
		**/
		float samplesPerNode = 1.5f;

		
		//! TODO(Islam): add documentation
		float lowDepthCutOff = 0.f;

		//! The importance that interpolation of the point samples is given in the formulation of the screened Poisson equation.
		/** The results of the original (unscreened) Poisson Reconstruction can be obtained by setting this value to 0.
		**/
		float pointWeight = 2.0f;

		//! The number of solver iterations
		/** Number of Gauss-Seidel relaxations to be performed at each level of the octree hierarchy.
		**/
		int iters = 8;

		//! If this flag is enabled, the sampling density is written out with the vertices
		bool withDensity = false;

		bool withExactInterpolation = false;

		//! This flag tells the reconstructor to read in color values with the input points and extrapolate those to the vertices of the output.
		bool withColors = true;

		//! use provide an enevelope for construction
		bool withEnvelope = false;
		char* envelopeFilePath = nullptr;
		bool withNoDirichletErode = true;;

		//! Data pull factor
		/** If withColors is true, this floating point value specifies the relative importance of finer color estimates over lower ones.
		**/
		float colorPullFactor = 32.0f;

		//! Normal confidence exponent
		/** Exponent to be applied to a point's confidence to adjust its weight. (A point's confidence is defined by the magnitude of its normal.)
		**/
		float normalConfidence = 0.0;
		
		//! Normal confidence bias exponent
		/** Exponent to be applied to a point's confidence to bias the resolution at which the sample contributes to the linear system. (Points with lower confidence are biased to contribute at coarser resolutions.)
		**/
		float normalConfidenceBias = 0.0;

		//! Enabling this flag has the reconstructor use linear interpolation to estimate the positions of iso-vertices.
		bool linearFit = false;

		//! This parameter specifies the number of threads across which the solver should be parallelized
		int threads = 1;

		/** The parameters below are accessible via the command line but are not described in the official documentation **/

		//! The depth beyond which the octree will be adapted.
		/** At coarser depths, the octree will be complete, containing all 2^d x 2^d x 2^d nodes.
		**/
		int fullDepth = 5;

		//! Coarse MG solver depth
		int baseDepth = 0;

		//! Coarse MG solver v-cycles
		int baseVCycles = 1;

		//! This flag specifies the accuracy cut-off to be used for CG
		float cgAccuracy = 1.0e-3f;

		bool verbose = false;
	};

	//! Input cloud interface
	template <typename Real> class ICloud
	{
	public:
		virtual size_t size() const = 0;
		virtual bool hasNormals() const = 0;
		virtual bool hasColors() const = 0;
		virtual void getPoint(size_t index, Real* coords) const = 0;
		virtual void getNormal(size_t index, Real* coords) const = 0;
		virtual void getColor(size_t index, Real* rgb) const = 0;
	};

	//! Output mesh interface
	template <typename Real> class IMesh
	{
	public:
		virtual void addVertex(const Real* coords) = 0;
		virtual void addNormal(const Real* coords) = 0;
		virtual void addColor(const Real* rgb) = 0;
		virtual void addDensity(double d) = 0;
		virtual void addTriangle(size_t i1, size_t i2, size_t i3) = 0;
	};

	//! Reconstruct a mesh from a point cloud (float version)
	static bool Reconstruct(Parameters& params,
							const PoissonReconLib::ICloud<float>& inCloud,
							PoissonReconLib::IMesh<float>& ouMesh);
};
