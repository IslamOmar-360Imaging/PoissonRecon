//##########################################################################
//#                                                                        #
//#               CLOUDCOMPARE WRAPPER: PoissonReconLib                    #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#               COPYRIGHT: Daniel Girardeau-Montaut                      #
//#                                                                        #
//##########################################################################

#include "PoissonReconLib.h"

// PoissonRecon
#include <FEMTree.h>
#include <PPolynomial.h>
#include <Ply.h>

// workaround for gcc
#define offsetof(s, m) ((::size_t) & reinterpret_cast<char const volatile&>((((s*)0)->m)))
#include <VertexFactory.h>

#include "PointData.h"

#include <cassert>
#include <limits>

MessageWriter messageWriter;

namespace
{
	// The order of the B-Spline used to splat in data for color interpolation
	constexpr int DATA_DEGREE = 0;
	// The order of the B-Spline used to splat in the weights for density estimation
	constexpr int WEIGHT_DEGREE = 2;
	// The order of the B-Spline used to splat in the normals for constructing the Laplacian
	// constraints
	constexpr int NORMAL_DEGREE = 2;
	// The default finite-element degree
	constexpr int DEFAULT_FEM_DEGREE = 1;
	// The dimension of the system
	constexpr int DIMENSION = 3;

	inline float
	ComputeNorm(const float vec[3])
	{
		return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
	}

	inline double
	ComputeNorm(const double vec[3])
	{
		return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
	}

	double
	Weight(double v, double start, double end)
	{
		v = (v - start) / (end - start);
		if (v < 0)
			return 1.;
		else if (v > 1)
			return 0.;
		else
		{
			// P(x) = a x^3 + b x^2 + c x + d
			//		P (0) = 1 , P (1) = 0 , P'(0) = 0 , P'(1) = 0
			// =>	d = 1 , a + b + c + d = 0 , c = 0 , 3a + 2b + c = 0
			// =>	c = 0 , d = 1 , a + b = -1 , 3a + 2b = 0
			// =>	a = 2 , b = -3 , c = 0 , d = 1
			// =>	P(x) = 2 x^3 - 3 x^2 + 1
			return 2. * v * v * v - 3. * v * v + 1.;
		}
	}
} // namespace

PoissonReconLib::Parameters::Parameters()
{
#ifdef WITH_OPENMP
	threads = omp_get_num_procs();
#endif
}

template <typename _Real>
class Vertex : public PointData<_Real>
{
public:
	typedef _Real Real;

	Vertex(const Point<Real, 3>& point)
		: PointData<Real>()
		, point(point)
		, w(0)
	{
	}

	Vertex(const Point<Real, 3>& point, const PointData<Real>& data, double _w = 0.0)
		: PointData<Real>(data.normal, data.color)
		, point(point)
		, w(_w)
	{
	}

	Vertex()
		: Vertex(Point<Real, 3>(0, 0, 0))
	{
	}

	Vertex&
	operator*=(Real s)
	{
		PointData<Real>::operator*=(s);
		point *= s;
		w *= s;

		return *this;
	}

	Vertex&
	operator/=(Real s)
	{
		PointData<Real>::operator*=(1 / s);
		point /= s;
		w /= s;
		return *this;
	}

	Vertex&
	operator+=(const Vertex& p)
	{
		PointData<Real>::operator+=(p);
		point += p.point;
		w += p.w;

		return *this;
	}

	template <size_t N>
	Point<Real, 3>&
	get()
	{
		return point;
	}

	template <size_t N>
	const Point<Real, 3>
	get() const
	{
		return point;
	}

public:
	Point<Real, 3> point;
	double w;
};

template <typename Real, unsigned int Dim, typename FunctionValueType = PointData<Real>>
using InputPointStreamInfo =
	typename FEMTreeInitializer<Dim, Real>::template InputPointStream<FunctionValueType>;

template <typename Real, unsigned int Dim, typename FunctionValueType = PointData<Real>>
using InputPointStream =
	typename InputPointStreamInfo<Real, Dim, FunctionValueType>::StreamType;

template <typename Real, unsigned int Dim, typename FunctionValueType = PointData<Real>>
using InputPointStreamWithData = InputPointStream<Real, Dim, FunctionValueType>;

// template <typename Real, unsigned int dim>
// using InputPointStream = typename FEMTreeInitializer<dim,Real>::template
// InputPointStream<PointData<Real>>::StreamType;

template <typename Real>
class PointStream : public InputPointStreamWithData<Real, DIMENSION, PointData<Real>>
{
public:
	PointStream(const PoissonReconLib::ICloud<Real>& _cloud)
		: cloud(_cloud)
		, xform(nullptr)
		, currentIndex(0)
	{
	}

	void
	reset(void) override
	{
		currentIndex = 0;
	}

	bool
	next(typename InputPointStreamInfo<Real, DIMENSION, PointData<Real>>::PointAndDataType&
				 x) override
	{
		if (currentIndex >= cloud.size())
		{
			return false;
		}

		auto p = x.template get<0>();
		cloud.getPoint(currentIndex, p.coords);

		if (xform != nullptr)
		{
			p = (*xform) * p;
		}
		auto& d = x.template get<1>();
		if (cloud.hasNormals())
		{
			cloud.getNormal(currentIndex, d.normal);
		}
		else
		{
			d.normal[0] = d.normal[1] = d.normal[2];
		}

		if (cloud.hasColors())
		{
			cloud.getColor(currentIndex, d.color);
		}
		else
		{
			d.color[0] = d.color[1] = d.color[2];
		}

		currentIndex++;
		return true;
	}

public:
	const PoissonReconLib::ICloud<Real>& cloud;
	XForm<Real, 4>* xform;
	size_t currentIndex;
};

template< unsigned int Dim , class Real >
struct FEMTreeProfiler
{
	double t;

	void start( void ){ t = Time() , FEMTree< Dim , Real >::ResetLocalMemoryUsage(); }
	void print( const char* header ) const
	{
		FEMTree< Dim , Real >::MemoryUsage();
		if( header ) printf( "%s %9.1f (s), %9.1f (MB) / %9.1f (MB) / %d (MB)\n" , header , Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
		else         printf(    "%9.1f (s), %9.1f (MB) / %9.1f (MB) / %d (MB)\n" ,          Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
	}
	void dumpOutput( const char* header ) const
	{
		FEMTree< Dim , Real >::MemoryUsage();
		if( header ) messageWriter( "%s %9.1f (s), %9.1f (MB) / %9.1f (MB) / %d (MB)\n" , header , Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
		else         messageWriter(    "%9.1f (s), %9.1f (MB) / %9.1f (MB) / %d (MB)\n" ,          Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
	}
	void dumpOutput2( std::vector< std::string >& comments , const char* header ) const
	{
		FEMTree< Dim , Real >::MemoryUsage();
		if( header ) messageWriter( comments , "%s %9.1f (s), %9.1f (MB) / %9.1f (MB) / %d (MB)\n" , header , Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
		else         messageWriter( comments ,    "%9.1f (s), %9.1f (MB) / %9.1f (MB) / %d (MB)\n" ,          Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
	}
};
template< class Real , unsigned int Dim >
XForm< Real , Dim+1 > GetBoundingBoxXForm( Point< Real , Dim > min , Point< Real , Dim > max , Real scaleFactor )
{
	Point< Real , Dim > center = ( max + min ) / 2;
	Real scale = max[0] - min[0];
	for( int d=1 ; d<Dim ; d++ ) scale = std::max< Real >( scale , max[d]-min[d] );
	scale *= scaleFactor;
	for( int i=0 ; i<Dim ; i++ ) center[i] -= scale/2;
	XForm< Real , Dim+1 > tXForm = XForm< Real , Dim+1 >::Identity() , sXForm = XForm< Real , Dim+1 >::Identity();
	for( int i=0 ; i<Dim ; i++ ) sXForm(i,i) = (Real)(1./scale ) , tXForm(Dim,i) = -center[i];
	return sXForm * tXForm;
}
template< class Real , unsigned int Dim >
XForm< Real , Dim+1 > GetBoundingBoxXForm( Point< Real , Dim > min , Point< Real , Dim > max , Real width , Real scaleFactor , int& depth )
{
	// Get the target resolution (along the largest dimension)
	Real resolution = ( max[0]-min[0] ) / width;
	for( int d=1 ; d<Dim ; d++ ) resolution = std::max< Real >( resolution , ( max[d]-min[d] ) / width );
	resolution *= scaleFactor;
	depth = 0;
	while( (1<<depth)<resolution ) depth++;

	Point< Real , Dim > center = ( max + min ) / 2;
	Real scale = (1<<depth) * width;

	for( int i=0 ; i<Dim ; i++ ) center[i] -= scale/2;
	XForm< Real , Dim+1 > tXForm = XForm< Real , Dim+1 >::Identity() , sXForm = XForm< Real , Dim+1 >::Identity();
	for( int i=0 ; i<Dim ; i++ ) sXForm(i,i) = (Real)(1./scale ) , tXForm(Dim,i) = -center[i];
	return sXForm * tXForm;
}

template <class Real, unsigned int Dim>
XForm<Real, Dim + 1>
GetPointXForm(
	InputPointStream<Real, Dim>& stream,
	Real width,
	Real scaleFactor,
	int& depth)
{
	Point<Real, Dim> min, max;
	InputPointStreamInfo<Real, Dim>::BoundingBox(stream, min, max);
	return GetBoundingBoxXForm(min, max, width, scaleFactor, depth);
}

template <class Real, unsigned int Dim>
XForm<Real, Dim + 1>
GetPointXForm(InputPointStream<Real, Dim>& stream, Real scaleFactor)
{
	Point<Real, Dim> min, max;
	InputPointStreamInfo<Real, Dim>::BoundingBox(stream, min, max);
	return GetBoundingBoxXForm(min, max, scaleFactor);
}


template< unsigned int Dim , typename Real >
struct ConstraintDual
{
	Real target , weight;
	ConstraintDual( Real t , Real w ) : target(t) , weight(w){ }
	CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim >& p ) const { return CumulativeDerivativeValues< Real , Dim , 0 >( target*weight ); };
};
template< unsigned int Dim , typename Real >
struct SystemDual
{
	Real weight;
	SystemDual( Real w ) : weight(w){ }
	CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim >& p , const CumulativeDerivativeValues< Real , Dim , 0 >& dValues ) const { return dValues * weight; };
	CumulativeDerivativeValues< double , Dim , 0 > operator()( const Point< Real , Dim >& p , const CumulativeDerivativeValues< double , Dim , 0 >& dValues ) const { return dValues * weight; };
};
template< unsigned int Dim >
struct SystemDual< Dim , double >
{
	typedef double Real;
	Real weight;
	SystemDual( Real w ) : weight(w){ }
	CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim >& p , const CumulativeDerivativeValues< Real , Dim , 0 >& dValues ) const { return dValues * weight; };
};

template <
	typename Vertex,
	typename Real,
	typename SetVertexFunction,
	unsigned int... FEMSigs,
	typename... SampleData>
void
ExtractMesh(
	const PoissonReconLib::Parameters& params,
	UIntPack<FEMSigs...>,
	std::tuple<SampleData...>,
	FEMTree<sizeof...(FEMSigs), Real>& tree,
	const DenseNodeData<Real, UIntPack<FEMSigs...>>& solution,
	Real isoValue,
	const std::vector<typename FEMTree<sizeof...(FEMSigs), Real>::PointSample>* samples,
	std::vector<PointData<Real>>* sampleData,
	const typename FEMTree<sizeof...(FEMSigs), Real>::template DensityEstimator<
		WEIGHT_DEGREE>* density,
	const SetVertexFunction& SetVertex,
	XForm<Real, sizeof...(FEMSigs) + 1> iXForm,
	PoissonReconLib::IMesh<Real>& out_mesh)
{
	static const int Dim = sizeof...(FEMSigs);
	typedef UIntPack<FEMSigs...> Sigs;
	static const unsigned int DataSig =
		FEMDegreeAndBType<DATA_DEGREE, BOUNDARY_FREE>::Signature;

	const bool non_manifold = true;
	const bool polygon_mesh = false;

	CoredVectorMeshData<Vertex, node_index_type> mesh;

	typename IsoSurfaceExtractor<Dim, Real, Vertex>::IsoStats isoStats;

	if (samples && sampleData)
	{
		typedef typename FEMTree<Dim, Real>::template DensityEstimator<WEIGHT_DEGREE>
			DensityEstimator;

		SparseNodeData<ProjectiveData<PointData<Real>, Real>, IsotropicUIntPack<Dim, DataSig>>
			_sampleData = tree.template setExtrapolatedDataField<DataSig, false>(
				*samples,
				*sampleData,
				(DensityEstimator*)nullptr);

		for (const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* n =
					 tree.tree().nextNode();
				 n;
				 n = tree.tree().nextNode(n))
		{
			ProjectiveData<PointData<Real>, Real>* color = _sampleData(n);
			if (color)
				(*color) *= static_cast<Real>(pow(params.colorPullFactor, tree.depth(n)));
		}

		isoStats =IsoSurfaceExtractor<Dim, Real, Vertex>::template Extract<PointData<Real>>(
			Sigs(),
			UIntPack<WEIGHT_DEGREE>(),
			UIntPack<DataSig>(),
			tree,
			density,
			&_sampleData,
			solution,
			isoValue,
			mesh,
			PointData<Real>(),
			SetVertex,
			!params.linearFit,
			false,
			!non_manifold,
			polygon_mesh,
			false);
	}
	else
	{
		isoStats =IsoSurfaceExtractor<Dim, Real, Vertex>::template Extract<PointData<Real>>(
			Sigs(),
			UIntPack<WEIGHT_DEGREE>(),
			UIntPack<DataSig>(),
			tree,
			density,
			nullptr,
			solution,
			isoValue,
			mesh,
			PointData<Real>(),
			SetVertex,
			!params.linearFit,
			false,
			!non_manifold,
			polygon_mesh,
			false);
	}

	mesh.resetIterator();

	for (size_t vidx = 0; vidx < mesh.outOfCoreVertexNum(); ++vidx)
	{
		Vertex v;
		mesh.nextOutOfCoreVertex(v);
		v.point = iXForm * v.point;

		out_mesh.addVertex(v.point.coords);
		if (sampleData)
		{
			// out_mesh.addNormal(v.normal);
			out_mesh.addColor(v.color);
		}
		if (params.withDensity)
		{
			out_mesh.addDensity(v.w);
		}
	}

	for (size_t tidx = 0; tidx < mesh.polygonNum(); ++tidx)
	{
		std::vector<CoredVertexIndex<node_index_type>> triangle;
		mesh.nextPolygon(triangle);
		if (triangle.size() == 3)
		{
			out_mesh.addTriangle(triangle[0].idx, triangle[1].idx, triangle[2].idx);
		}
		else
		{
			assert(false);
		}
	}
}

// workhorse
template <class Real, typename... SampleData, unsigned int... FEMSigs>
static bool
Execute(
	PointStream<Real>& pointStream,
	PoissonReconLib::IMesh<Real>& out_mesh,
  PoissonReconLib::Parameters& params,
	UIntPack<FEMSigs...>)
{
	static const int Dim = sizeof ... ( FEMSigs );
	typedef UIntPack< FEMSigs ... > Sigs;
	typedef UIntPack< FEMSignature< FEMSigs >::Degree ... > Degrees;
	typedef UIntPack< FEMDegreeAndBType< NORMAL_DEGREE , DerivativeBoundary< FEMSignature< FEMSigs >::BType , 1 >::BType >::Signature ... > NormalSigs;
	static const unsigned int DataSig = FEMDegreeAndBType< DATA_DEGREE , BOUNDARY_FREE >::Signature;
	typedef typename FEMTree< Dim , Real >::template DensityEstimator< WEIGHT_DEGREE > DensityEstimator;
	typedef typename FEMTree< Dim , Real >::template InterpolationInfo< Real , 0 > InterpolationInfo;


	typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > FEMTreeNode;
	typedef typename FEMTreeInitializer< Dim , Real >::GeometryNodeType GeometryNodeType;

	// Compute scaling transformation (and optionally the depth)
	XForm<Real, Dim + 1> modelToUnitCube, unitCubeToModel;
	modelToUnitCube = XForm<Real, Dim + 1>::Identity();
	{
		if (params.finestCellWidth > 0)
		{
			Real scaleFactor = static_cast<Real>(params.scale > 0 ? params.scale : 1.0);
			modelToUnitCube = GetPointXForm<Real, Dim>(
								pointStream,
								params.finestCellWidth,
								scaleFactor,
								params.depth) *
				modelToUnitCube; // warning: depth may change!
		}
		else if (params.scale > 0)
		{
			modelToUnitCube =
				GetPointXForm<Real, Dim>(pointStream, static_cast<Real>(params.scale)) * modelToUnitCube;
		}
		pointStream.xform = &modelToUnitCube;
	}
	Real isoValue = 0;

	FEMTree< Dim , Real > tree( MEMORY_ALLOCATOR_BLOCK_SIZE );
	FEMTreeProfiler< Dim , Real > profiler;

	if (params.depth > 0 && params.finestCellWidth > 0)
	{
		params.finestCellWidth = 0;
	}

	size_t pointCount;

	Real pointWeightSum;
	std::vector< typename FEMTree< Dim , Real >::PointSample >* samples = new std::vector< typename FEMTree< Dim , Real >::PointSample >();
	typedef PointData<Real> InputSampleDataType;
	typedef std::vector<PointData<Real>> SampleDataSet;
	DenseNodeData< GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > geometryNodeDesignators;
	SampleDataSet* sampleData = NULL;
	DensityEstimator* density = NULL;
	SparseNodeData< Point< Real , Dim > , NormalSigs >* normalInfo = NULL;
	Real targetValue = (Real)0.5;

	// Read in the samples (and color data)
	{
		profiler.start();
		if( params.finestCellWidth>0 )
		{
			if( params.solveDepth == 0 ) params.solveDepth = params.depth;
			if( params.solveDepth>params.depth )
			{
				WARN( "Solution depth cannot exceed system depth: " , params.solveDepth, " <= " , params.depth );
				params.solveDepth = params.depth;
			}
			if (params.fullDepth > params.depth)
			{
				WARN( "Full depth cannot exceed system depth: " , params.fullDepth , " <= " , params.depth );
				params.fullDepth = params.depth;
			}
			if (params.baseDepth > params.fullDepth)
			{
				WARN(
					"Base depth must be smaller than full depth: ",
					params.baseDepth,
					" <= ",
					params.fullDepth);
				params.baseDepth = params.fullDepth;
			}
		}

		{
			auto ProcessDataWithConfidence = [&]( const Point< Real , Dim > &p , typename InputPointStreamInfo<Real,Dim>::DataType &d )
			{
				Real l = ComputeNorm(d.normal);
				if( !l || !std::isfinite( l ) ) return (Real)-1.;
				return (Real)pow( l , params.normalConfidence );
			};
			auto ProcessData = []( const Point< Real , Dim > &p , typename InputPointStreamInfo<Real,Dim>::DataType &d )
			{
				Real l = ComputeNorm(d.normal);
				if( !l || !std::isfinite( l ) ) return (Real)-1.;
			  d.normal[0] /= l;
				d.normal[1] /= l;
				d.normal[2] /= l;
				return (Real)1.;
			};
			if( params.normalConfidence>0 ) pointCount = FEMTreeInitializer< Dim , Real >::template Initialize< InputSampleDataType >( tree.spaceRoot() , pointStream , params.depth , *samples , *sampleData , true , tree.nodeAllocators[0] , tree.initializer() , ProcessDataWithConfidence );
			else                     pointCount = FEMTreeInitializer< Dim , Real >::template Initialize< InputSampleDataType >( tree.spaceRoot() , pointStream , params.depth , *samples , *sampleData , true , tree.nodeAllocators[0] , tree.initializer() , ProcessData );
		}

		unitCubeToModel = modelToUnitCube.inverse();
	}

	DenseNodeData< Real , Sigs > solution;
	{
		DenseNodeData< Real , Sigs > constraints;
		InterpolationInfo* iInfo = NULL;
		int solveDepth = params.depth;

		tree.resetNodeIndices( 0 , std::make_tuple() );

		// Get the kernel density estimator
		{
			profiler.start();
			density = tree.template setDensityEstimator<1, WEIGHT_DEGREE>(
				*samples,
				params.kernelDepth,
				params.samplesPerNode);
		}

		// Transform the Hermite samples into a vector field
		{
			profiler.start();
			normalInfo = new SparseNodeData< Point< Real , Dim > , NormalSigs >();
			std::function< bool ( InputSampleDataType , Point< Real , Dim >& ) > ConversionFunction = []( InputSampleDataType in , Point< Real , Dim > &out )
			{
				Point< Real , Dim > n(in.normal[0], in.normal[1], in.normal[2]);
				Real l = (Real)Length( n );
				// It is possible that the samples have non-zero normals but there are two co-located samples with negative normals...
				if( !l ) return false;
				out = n / l;
				return true;
			};
			std::function< bool ( InputSampleDataType , Point< Real , Dim >& , Real & ) > ConversionAndBiasFunction = [&]( InputSampleDataType in , Point< Real , Dim > &out , Real &bias )
			{
				Point<Real, Dim> n(in.normal[0], in.normal[1], in.normal[2]);;
				Real l = (Real)Length( n );
				// It is possible that the samples have non-zero normals but there are two co-located samples with negative normals...
				if( !l ) return false;
				out = n / l;
				bias = (Real)( log( l ) *  params.normalConfidenceBias/ log( 1<<(Dim-1) ) );
				return true;
			};

			if( params.normalConfidenceBias>0 ) *normalInfo = tree.setInterpolatedDataField( NormalSigs() , *samples , *sampleData , density , params.baseDepth , params.depth , (Real)params.lowDepthCutOff , pointWeightSum , ConversionAndBiasFunction );
			else
				*normalInfo = tree.setInterpolatedDataField(
					NormalSigs(),
					*samples,
					*sampleData,
					density,
					params.baseDepth,
					params.depth,
					(Real)params.lowDepthCutOff,
					pointWeightSum,
					ConversionFunction);
			ThreadPool::Parallel_for( 0 , normalInfo->size() , [&]( unsigned int , size_t i ){ (*normalInfo)[i] *= (Real)-1.; } );
		}

		// Get the geometry designators indicating if the space node are interior to, exterior to, or contain the envelope boundary
		if( params.withEnvelope )
		{
			profiler.start();
			{
				// Make the octree complete up to the base depth
				std::function< void ( FEMTreeNode * , unsigned int ) > MakeComplete = [&]( FEMTreeNode *node , unsigned int depth )
				{
					if( node->depth()<(int)depth )
					{
						if( !node->children ) node->template initChildren< false >( tree.nodeAllocators.size() ?  tree.nodeAllocators[0] : NULL , tree.initializer() );
						for( int c=0 ; c<(1<<Dim) ; c++ ) MakeComplete( node->children+c , depth );
					}
				};
				MakeComplete( &tree.spaceRoot() , params.baseDepth);

				// Read in the envelope geometry
				std::vector< Point< Real , Dim > > vertices;
				std::vector< SimplexIndex< Dim-1 , node_index_type > > simplices;
				{
					using namespace VertexFactory;
					std::vector<typename PositionFactory<Real, Dim>::VertexType> _vertices;
					std::vector< std::vector< int > > polygons;
					std::vector< std::string > comments;
					int file_type;
					PLY::ReadPolygons(
						std::string(params.envelopeFilePath),
						PositionFactory< Real , Dim >(),
						_vertices,
						polygons,
						file_type,
						comments);
					vertices.resize( _vertices.size() );
					for( int i=0 ; i<vertices.size() ; i++ ) vertices[i] = modelToUnitCube * _vertices[i];
					simplices.resize( polygons.size() );
					for( int i=0 ; i<polygons.size() ; i++ )
						if( polygons[i].size()!=Dim ) ERROR_OUT( "Not a simplex" );
						else for( int j=0 ; j<Dim ; j++ ) simplices[i][j] = polygons[i][j];
				}
				// Get the interior/boundary/exterior designators
				geometryNodeDesignators = FEMTreeInitializer< Dim , Real >::template GetGeometryNodeDesignators( &tree.spaceRoot() , vertices , simplices , params.baseDepth , params.envelopeDepth , tree.nodeAllocators , tree.initializer() );

				// Make nodes in the support of the vector field @{ExactDepth} interior
				if( params.withNoDirichletErode)
				{
					// What to do if we find a node in the support of the vector field
					auto SetScratchFlag = [&]( FEMTreeNode *node )
					{
						if( node )
						{
							while( node->depth()>params.baseDepth ) node = node->parent;
							node->nodeData.setScratchFlag( true );
						}
					};

					std::function< void ( FEMTreeNode * ) > PropagateToLeaves = [&]( const FEMTreeNode *node )
					{
						geometryNodeDesignators[ node ] = GeometryNodeType::INTERIOR;
						if( node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) PropagateToLeaves( node->children+c );
					};

					// Flags indicating if a node contains a non-zero vector field coefficient
					std::vector< bool > isVectorFieldElement( tree.nodeCount() , false );

					// Get the set of base nodes
					std::vector< FEMTreeNode * > baseNodes;
					auto TerminationLambda = [&]( const FEMTreeNode *node ){ return node->depth()==params.baseDepth; };
					for( FEMTreeNode *node=tree.spaceRoot().nextNode( TerminationLambda , NULL ) ; node ; node=tree.spaceRoot().nextNode( TerminationLambda , node ) ) if( node->depth()==params.baseDepth ) baseNodes.push_back( node );

					std::vector< node_index_type > vectorFieldElementCounts( baseNodes.size() );
					for( int i=0 ; i<vectorFieldElementCounts.size() ; i++ ) vectorFieldElementCounts[i] = 0;

					// In parallel, iterate over the base nodes and mark the nodes containing non-zero vector field coefficients
					ThreadPool::Parallel_for( 0 , baseNodes.size() , [&]( unsigned int t , size_t  i )
					{
						for( FEMTreeNode *node=baseNodes[i]->nextNode() ; node ; node=baseNodes[i]->nextNode(node) )
						{
							Point< Real , Dim > *n = (*normalInfo)( node );
							if( n && Point< Real , Dim >::SquareNorm( *n ) ) isVectorFieldElement[ node->nodeData.nodeIndex ] = true , vectorFieldElementCounts[i]++;
						}
					} );
					size_t vectorFieldElementCount = 0;
					for( int i=0 ; i<vectorFieldElementCounts.size() ; i++ ) vectorFieldElementCount += vectorFieldElementCounts[i];

					// Get the subset of nodes containing non-zero vector field coefficients and disable the "scratch" flag
					std::vector< FEMTreeNode * > vectorFieldElements;
					vectorFieldElements.reserve( vectorFieldElementCount );
					{
						std::vector< std::vector< FEMTreeNode * > > _vectorFieldElements( baseNodes.size() );
						for( int i=0 ; i<_vectorFieldElements.size() ; i++ ) _vectorFieldElements[i].reserve( vectorFieldElementCounts[i] );
						ThreadPool::Parallel_for( 0 , baseNodes.size() , [&]( unsigned int t , size_t  i )
						{
							for( FEMTreeNode *node=baseNodes[i]->nextNode() ; node ; node=baseNodes[i]->nextNode(node) )
							{
								if( isVectorFieldElement[ node->nodeData.nodeIndex ] ) _vectorFieldElements[i].push_back( node );
								node->nodeData.setScratchFlag( false );
							}
						} );
						for( int i=0 ; i<_vectorFieldElements.size() ; i++ ) vectorFieldElements.insert( vectorFieldElements.end() , _vectorFieldElements[i].begin() , _vectorFieldElements[i].end() );
					}

					// Set the scratch flag for the base nodes on which the vector field is supported
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] In principal, we should unlock finite elements whose support overlaps the vector field" )
#endif // SHOW_WARNINGS
					tree.template processNeighboringLeaves< -BSplineSupportSizes< NORMAL_DEGREE >::SupportStart , BSplineSupportSizes< NORMAL_DEGREE >::SupportEnd >( &vectorFieldElements[0] , vectorFieldElements.size() , SetScratchFlag , false );

					// Set sub-trees rooted at interior nodes @ ExactDepth to interior
					ThreadPool::Parallel_for( 0 , baseNodes.size() , [&]( unsigned int , size_t  i ){ if( baseNodes[i]->nodeData.getScratchFlag() ) PropagateToLeaves( baseNodes[i] ); } );

					// Adjust the coarser node designators in case exterior nodes have become boundary.
					ThreadPool::Parallel_for( 0 , baseNodes.size() , [&]( unsigned int , size_t  i ){ FEMTreeInitializer< Dim , Real >::PullGeometryNodeDesignatorsFromFiner( baseNodes[i] , geometryNodeDesignators ); } );
					FEMTreeInitializer< Dim , Real >::PullGeometryNodeDesignatorsFromFiner( &tree.spaceRoot() , geometryNodeDesignators , params.baseDepth );
				}
			}
		}

		if( !params.withDensity ) delete density , density = NULL;

		// Add the interpolation constraints
		if( params.pointWeight>0 )
		{
			profiler.start();
			if (params.withExactInterpolation)
				iInfo =
					FEMTree<Dim, Real>::template InitializeExactPointInterpolationInfo<Real, 0>(
						tree,
						*samples,
						ConstraintDual<Dim, Real>(
							targetValue,
							(Real)params.pointWeight * pointWeightSum),
						SystemDual<Dim, Real>((Real)params.pointWeight * pointWeightSum),
						true,
						false);
			else
				iInfo = FEMTree<Dim, Real>::
					template InitializeApproximatePointInterpolationInfo<Real, 0>(
						tree,
						*samples,
						ConstraintDual<Dim, Real>(
							targetValue,
							(Real)params.pointWeight * pointWeightSum),
						SystemDual<Dim, Real>((Real)params.pointWeight * pointWeightSum),
						true,
						1);
		}

		// Trim the tree and prepare for multigrid
		{
			profiler.start();
			constexpr int MAX_DEGREE = NORMAL_DEGREE > Degrees::Max() ? NORMAL_DEGREE : Degrees::Max();
			typename FEMTree< Dim , Real >::template HasNormalDataFunctor< NormalSigs > hasNormalDataFunctor( *normalInfo );
			auto hasDataFunctor = [&]( const FEMTreeNode *node ){ return hasNormalDataFunctor( node ); };
			if( geometryNodeDesignators.size() ) tree.template finalizeForMultigrid< MAX_DEGREE , Degrees::Max() >( params.baseDepth , params.fullDepth , hasDataFunctor , [&]( const FEMTreeNode *node ){ return node->nodeData.nodeIndex<(node_index_type)geometryNodeDesignators.size() && geometryNodeDesignators[node]==GeometryNodeType::EXTERIOR; } , std::make_tuple( iInfo ) , std::make_tuple( normalInfo , density , &geometryNodeDesignators ) );
			else                                 tree.template finalizeForMultigrid< MAX_DEGREE , Degrees::Max() >( params.baseDepth , params.fullDepth , hasDataFunctor , []( const FEMTreeNode * ){ return false; } , std::make_tuple( iInfo ) , std::make_tuple( normalInfo , density ) );

		}
		// Add the FEM constraints
		{
			profiler.start();
			constraints = tree.initDenseNodeData( Sigs() );
			typename FEMIntegrator::template Constraint< Sigs , IsotropicUIntPack< Dim , 1 > , NormalSigs , IsotropicUIntPack< Dim , 0 > , Dim > F;
			unsigned int derivatives2[Dim];
			for( int d=0 ; d<Dim ; d++ ) derivatives2[d] = 0;
			typedef IsotropicUIntPack< Dim , 1 > Derivatives1;
			typedef IsotropicUIntPack< Dim , 0 > Derivatives2;
			for( int d=0 ; d<Dim ; d++ )
			{
				unsigned int derivatives1[Dim];
				for( int dd=0 ; dd<Dim ; dd++ ) derivatives1[dd] = dd==d ?  1 : 0;
				F.weights[d][ TensorDerivatives< Derivatives1 >::Index( derivatives1 ) ][ TensorDerivatives< Derivatives2 >::Index( derivatives2 ) ] = 1;
			}
			tree.addFEMConstraints( F , *normalInfo , constraints , solveDepth );
		}

		// Free up the normal info
		delete normalInfo , normalInfo = NULL;

		// Add the interpolation constraints
		if( params.pointWeight>0 )
		{
			profiler.start();
			tree.addInterpolationConstraints( constraints , solveDepth , std::make_tuple( iInfo ) );
		}

		messageWriter( "Leaf Nodes / Active Nodes / Ghost Nodes / Dirichlet Supported Nodes: %llu / %llu / %llu / %llu\n" , (unsigned long long)tree.leaves() , (unsigned long long)tree.nodes() , (unsigned long long)tree.ghostNodes() , (unsigned long long)tree.dirichletElements() );
		messageWriter( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage())/(1<<20) );
		
		// Solve the linear system
		{
			profiler.start();
			typename FEMTree< Dim , Real >::SolverInfo sInfo;
			sInfo.cgDepth = 0 , sInfo.cascadic = true , sInfo.vCycles = 1 , sInfo.iters = params.iters , sInfo.cgAccuracy = params.cgAccuracy , sInfo.verbose = params.verbose , sInfo.showResidual = false , sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE , sInfo.sliceBlockSize = 1;
			sInfo.baseVCycles = params.baseVCycles;
			typename FEMIntegrator::template System< Sigs , IsotropicUIntPack< Dim , 1 > > F( { 0. , 1. } );
			solution = tree.solveSystem( Sigs() , F , constraints , params.solveDepth , sInfo , std::make_tuple( iInfo ) );
			if( iInfo ) delete iInfo , iInfo = NULL;
		}
	}

	{
		profiler.start();
		double valueSum = 0 , weightSum = 0;
		typename FEMTree< Dim , Real >::template MultiThreadedEvaluator< Sigs , 0 > evaluator( &tree , solution );
		std::vector< double > valueSums( ThreadPool::NumThreads() , 0 ) , weightSums( ThreadPool::NumThreads() , 0 );
		ThreadPool::Parallel_for( 0 , samples->size() , [&]( unsigned int thread , size_t j )
		{
			ProjectiveData< Point< Real , Dim > , Real >& sample = (*samples)[j].sample;
			Real w = sample.weight;
			if( w>0 ) weightSums[thread] += w , valueSums[thread] += evaluator.values( sample.data / sample.weight , thread , (*samples)[j].node )[0] * w;
		}
		);
		for( size_t t=0 ; t<valueSums.size() ; t++ ) valueSum += valueSums[t] , weightSum += weightSums[t];
		isoValue = (Real)( valueSum / weightSum );
		delete samples, samples = NULL;
		profiler.dumpOutput( "Got average:" );
		messageWriter( "Iso-Value: %e = %g / %g\n" , isoValue , valueSum , weightSum );
	}

	auto SetVertex = [](
										 Vertex<Real>& v,
										 Point<Real, Dim> p,
										 Point<Real, Dim> g,
										 double w,
										 PointData<Real> d) { v = Vertex<Real>(p, d, w); };

	ExtractMesh<Vertex<Real>, Real>(
		params,
		UIntPack<FEMSigs...>(),
		std::tuple<SampleData...>(),
		tree,
		solution,
		isoValue,
		samples,
		sampleData,
		density,
		SetVertex,
		unitCubeToModel.inverse(),
		out_mesh);
}

bool
PoissonReconLib::Reconstruct(
	Parameters& params,
	const ICloud<float>& inCloud,
	IMesh<float>& outMesh)
{
	if (!inCloud.hasNormals())
	{
		// we need normals
		return false;
	}

#ifdef WITH_OPENMP
	ThreadPool::Init(
		(ThreadPool::ParallelType)(int)ThreadPool::OPEN_MP,
		std::thread::hardware_concurrency());
#else
	ThreadPool::Init(
		(ThreadPool::ParallelType)(int)ThreadPool::THREAD_POOL,
		std::thread::hardware_concurrency());
#endif

	PointStream<float> pointStream(inCloud);

	bool success = false;

	switch (params.boundary)
	{
	case Parameters::FREE:
		typedef IsotropicUIntPack<
			DIMENSION,
			FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_FREE>::Signature>
			FEMSigsFree;
		success = Execute<float>(pointStream, outMesh, params, FEMSigsFree());
		break;
	case Parameters::DIRICHLET:
		typedef IsotropicUIntPack<
			DIMENSION,
			FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_DIRICHLET>::Signature>
			FEMSigsDirichlet;
		success = Execute<float>(pointStream, outMesh, params, FEMSigsDirichlet());
		break;
	case Parameters::NEUMANN:
		typedef IsotropicUIntPack<
			DIMENSION,
			FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_NEUMANN>::Signature>
			FEMSigsNeumann;
		success = Execute<float>(pointStream, outMesh, params, FEMSigsNeumann());
		break;
	default:
		assert(false);
		break;
	}

	ThreadPool::Terminate();

	return success;
}

bool
PoissonReconLib::Reconstruct(
	Parameters& params,
	const ICloud<double>& inCloud,
	IMesh<double>& outMesh)
{
	if (!inCloud.hasNormals())
	{
		// we need normals
		return false;
	}

#ifdef WITH_OPENMP
	ThreadPool::Init(
		(ThreadPool::ParallelType)(int)ThreadPool::OPEN_MP,
		std::thread::hardware_concurrency());
#else
	ThreadPool::Init(
		(ThreadPool::ParallelType)(int)ThreadPool::THREAD_POOL,
		std::thread::hardware_concurrency());
#endif

	PointStream<double> pointStream(inCloud);

	bool success = false;

	switch (params.boundary)
	{
	case Parameters::FREE:
		typedef IsotropicUIntPack<
			DIMENSION,
			FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_FREE>::Signature>
			FEMSigsFree;
		success = Execute<double>(pointStream, outMesh, params, FEMSigsFree());
		break;
	case Parameters::DIRICHLET:
		typedef IsotropicUIntPack<
			DIMENSION,
			FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_DIRICHLET>::Signature>
			FEMSigsDirichlet;
		success = Execute<double>(pointStream, outMesh, params, FEMSigsDirichlet());
		break;
	case Parameters::NEUMANN:
		typedef IsotropicUIntPack<
			DIMENSION,
			FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_NEUMANN>::Signature>
			FEMSigsNeumann;
		success = Execute<double>(pointStream, outMesh, params, FEMSigsNeumann());
		break;
	default:
		assert(false);
		break;
	}

	ThreadPool::Terminate();

	return success;
}
