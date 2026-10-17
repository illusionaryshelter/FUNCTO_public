// Candidate appearance evidence with original FP64 scalar operation order.
// There is no approximate connectivity, iteration cutoff, or CPU retry path.
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>
#include <cstdint>

namespace {
constexpr int THREADS = 128;
struct Geometry {
  const double *xyz, *normal, *spacing;
  const bool *valid, *entity;
  const int64_t *pixels, *lookup;
  int64_t height, width;
  double factor, agreement, radius;
};

__device__ double dot021(double x, double y, double z,
                         double a, double b, double c) {
  // The checked NumPy einsum path is (p0+p2)+p1, unlike its norm reduction.
  return __dadd_rn(__dadd_rn(__dmul_rn(x,a), __dmul_rn(z,c)), __dmul_rn(y,b));
}
__device__ bool eligible_pixel(const Geometry& g, int64_t ref, int64_t x, int64_t y) {
  if (x<0 || y<0 || x>=g.width || y>=g.height) return false;
  const int64_t id = g.lookup[y*g.width+x];
  if (id<0 || !g.valid[id] || !g.entity[id]) return false;
  const double dx=__dsub_rn(g.xyz[3*id],g.xyz[3*ref]);
  const double dy=__dsub_rn(g.xyz[3*id+1],g.xyz[3*ref+1]);
  const double dz=__dsub_rn(g.xyz[3*id+2],g.xyz[3*ref+2]);
  const double height=dot021(dx,dy,dz,g.normal[3*ref],g.normal[3*ref+1],g.normal[3*ref+2]);
  const double agreement=dot021(g.normal[3*id],g.normal[3*id+1],g.normal[3*id+2],
                               g.normal[3*ref],g.normal[3*ref+1],g.normal[3*ref+2]);
  const double square=__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz));
  return fabs(height)<=__dmul_rn(g.factor,g.spacing[ref]) &&
         agreement>=g.agreement && __dsqrt_rn(square)<=g.radius;
}

__global__ void plan_kernel(int64_t n, const double* uv, const int64_t* integer,
    const bool* flags, const int64_t* query_rows, Geometry g,
    int64_t left, int64_t top, int64_t right, int64_t bottom,
    int64_t fh, int64_t fw, double rx, double ry, double resolution_minus_one,
    int64_t missing_state, int64_t* states, int64_t* feature_xy,
    int64_t* corner_pixels, double* weights, bool* eligible, int64_t* low, int64_t* high) {
  const int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (row>=n) return;
  int64_t state=integer[2*row+1];
  const int64_t ref=integer[2*row];
  for (int k=0;k<4;++k) {
    weights[4*row+k]=0.; eligible[4*row+k]=false;
    feature_xy[8*row+2*k]=feature_xy[8*row+2*k+1]=0;
    corner_pixels[8*row+2*k]=corner_pixels[8*row+2*k+1]=0;
  }
  low[2*row]=low[2*row+1]=high[2*row]=high[2*row+1]=0;
  if (query_rows[row]<0) { states[row]=state; return; }
  const bool position=flags[8*row+3], plane=flags[8*row+4];
  const bool oriented=flags[8*row+5], entity=flags[8*row+6], known=flags[8*row+7];
  if (position&&!plane) state=5;
  if (position&&plane&&known&&!oriented) state=11;
  const bool orientation=oriented||!known;
  if (position&&plane&&orientation&&!entity) state=12;
  if (!(position&&plane&&orientation&&entity)) { states[row]=state; return; }
  state=13;
  if (missing_state>=0) { states[row]=missing_state; return; }
  const double x=__dsub_rn(uv[2*row],double(left));
  const double y=__dsub_rn(uv[2*row+1],double(top));
  if (!(isfinite(x)&&isfinite(y)&&x>=0&&y>=0&&x<=double(right-left-1)&&y<=double(bottom-top-1))) {
    states[row]=17; return;
  }
  const double gx=__ddiv_rn(__dmul_rn(__dmul_rn(x,rx),double(fw-1)),resolution_minus_one);
  const double gy=__ddiv_rn(__dmul_rn(__dmul_rn(y,ry),double(fh-1)),resolution_minus_one);
  if (!(isfinite(gx)&&isfinite(gy)&&gx>=0&&gy>=0&&gx<=double(fw-1)&&gy<=double(fh-1))) {
    states[row]=18; return;
  }
  const int64_t ix=int64_t(floor(gx)), iy=int64_t(floor(gy));
  const double fx=__dsub_rn(gx,double(ix)), fy=__dsub_rn(gy,double(iy));
  const int64_t px=g.pixels[2*ref], py=g.pixels[2*ref+1];
  int64_t lx=px,ly=py,hx=px,hy=py;
  // Preserve (0,0),(1,0),(0,1),(1,1), including zero-weight corners.
  for (int k=0;k<4;++k) {
    const int dx=k&1,dy=k>>1;
    const int64_t cx=ix+dx,cy=iy+dy;
    const double weight=__dmul_rn(dx?fx:__dsub_rn(1.,fx),dy?fy:__dsub_rn(1.,fy));
    const double ox=__dadd_rn(__ddiv_rn(__ddiv_rn(__dmul_rn(double(cx),resolution_minus_one),double(fw-1)),rx),double(left));
    const double oy=__dadd_rn(__ddiv_rn(__ddiv_rn(__dmul_rn(double(cy),resolution_minus_one),double(fh-1)),ry),double(top));
    const int64_t cx_pixel=__double2ll_rn(ox),cy_pixel=__double2ll_rn(oy);
    feature_xy[8*row+2*k]=cx; feature_xy[8*row+2*k+1]=cy;
    corner_pixels[8*row+2*k]=cx_pixel; corner_pixels[8*row+2*k+1]=cy_pixel;
    weights[4*row+k]=weight;
    const bool good=weight>0&&eligible_pixel(g,ref,cx_pixel,cy_pixel);
    eligible[4*row+k]=good;
    if (good) {
      lx=min(lx,cx_pixel);ly=min(ly,cy_pixel);hx=max(hx,cx_pixel);hy=max(hy,cy_pixel);
    }
  }
  low[2*row]=lx;low[2*row+1]=ly;high[2*row]=hx;high[2*row+1]=hy;
  states[row]=state;
}

__global__ void connect_kernel(const int64_t* integer, Geometry g,
    const int64_t* corners, const bool* eligible, const int64_t* rows,
    const int64_t* low, const int64_t* high, const int64_t* offsets,
    int* visited, int* queue, bool* connected) {
  // One independent pixel graph per block. The atomic queue visits each
  // eligible pixel once. Queue ordering is irrelevant to the final component.
  const int item=blockIdx.x, tid=threadIdx.x;
  const int64_t row=rows[item],ref=integer[2*row];
  const int64_t lx=low[2*item],ly=low[2*item+1];
  const int64_t width=high[2*item]-lx+1,height=high[2*item+1]-ly+1;
  const int cells=int(width*height);
  int* seen=visited+offsets[item];int* fifo=queue+offsets[item];
  for (int cell=tid;cell<cells;cell+=blockDim.x) {
    const int64_t x=lx+cell%width,y=ly+cell/width;
    seen[cell]=eligible_pixel(g,ref,x,y)?0:-1;
  }
  __shared__ int head,tail,batch;
  __syncthreads();
  if (tid==0) {
    head=0;tail=0;
    const int64_t x=g.pixels[2*ref]-lx,y=g.pixels[2*ref+1]-ly;
    if (x>=0&&y>=0&&x<width&&y<height&&seen[y*width+x]==0) {
      const int seed=int(y*width+x);seen[seed]=1;fifo[tail++]=seed;
    }
  }
  __syncthreads();
  while (true) {
    if (tid==0) batch=min(tail-head,int(blockDim.x));
    __syncthreads();
    if (batch==0) break;
    if (tid<batch) {
      const int cell=fifo[head+tid];const int64_t x=cell%width,y=cell/width;
      for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx) {
        const int64_t xx=x+dx,yy=y+dy;
        if ((dx||dy)&&xx>=0&&yy>=0&&xx<width&&yy<height) {
          const int next=int(yy*width+xx);
          if (atomicCAS(seen+next,0,1)==0) {
            const int position=atomicAdd(&tail,1);fifo[position]=next;
          }
        }
      }
    }
    __syncthreads();
    if (tid==0) head+=batch;
    __syncthreads();
  }
  if (tid<4) {
    const int64_t x=corners[8*row+2*tid]-lx,y=corners[8*row+2*tid+1]-ly;
    connected[4*row+tid]=eligible[4*row+tid]&&x>=0&&y>=0&&x<width&&y<height&&seen[y*width+x]==1;
  }
}

__global__ void finish_kernel(int64_t n,const int64_t* integer,const bool* flags,
    const int64_t* query_rows,int64_t* states,const int64_t* features,
    const int64_t* pixels,const double* weights,const bool* connected,
    const double* response,const double* references,const int64_t* offsets,
    const int64_t* observation_to_element,int64_t fh,int64_t fw,
    int64_t left,int64_t top,int64_t right,int64_t bottom,
    double* scores,double* utilities,int64_t* elements,bool* valid) {
  const int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (row>=n) return;
  // BEGIN preserve NumPy np.full(..., np.nan)'s positive quiet-NaN payload.
  // CUDA's CUDART_NAN is negative, which is equal_nan but not byte-identical.
  const double unknown=__longlong_as_double(0x7ff8000000000000ULL);
  scores[row]=unknown;utilities[row]=unknown;elements[row]=-1;valid[row]=false;
  // END missing evidence retains its original serialized representation.
  if (query_rows[row]<0||states[row]!=13) return;
  bool complete=true,any=false;
  for (int k=0;k<4;++k) {
    const int64_t x=pixels[8*row+2*k],y=pixels[8*row+2*k+1];
    const bool positive=weights[4*row+k]>0;
    const bool support=connected[4*row+k]&&x>=left&&x<right&&y>=top&&y<bottom;
    complete&=!positive||support;any|=positive&&support;
  }
  if (!complete) { states[row]=any?19:20;return; }
  const int64_t query=query_rows[row];
  double value=0.;
  for (int k=0;k<4;++k) {
    const double weight=weights[4*row+k];
    if (weight>0) {
      const int64_t x=features[8*row+2*k],y=features[8*row+2*k+1];
      value=__dadd_rn(value,__dmul_rn(double(response[(query*fh+y)*fw+x]),weight));
    }
  }
  scores[row]=value;valid[row]=true;states[row]=flags[8*row+7]?21:22;
  elements[row]=observation_to_element[integer[2*row]];
  const int64_t begin=offsets[query],end=offsets[query+1];
  if (begin==end||!isfinite(value)||elements[row]<0) return;
  int64_t l=begin,r=end;
  while (l<r) { const int64_t m=l+(r-l)/2;if (references[m]<value) l=m+1;else r=m; }
  const int64_t lower=l-begin;l=begin;r=end;
  while (l<r) { const int64_t m=l+(r-l)/2;if (references[m]<=value) l=m+1;else r=m; }
  utilities[row]=fmax(__dsub_rn(__ddiv_rn(double(lower+l-begin),double(end-begin)),1.),0.);
}

void checked(const at::Tensor& value, at::ScalarType dtype, int device) {
  TORCH_CHECK(value.is_cuda()&&value.get_device()==device&&value.scalar_type()==dtype&&value.is_contiguous(),
              "Evidence tensors must have the declared dtype, device and contiguous storage");
}
Geometry geometry(const std::vector<at::Tensor>& a,int start,const std::vector<double>& p) {
  const auto& lookup=a[start+6];
  return {a[start].data_ptr<double>(),a[start+1].data_ptr<double>(),a[start+2].data_ptr<double>(),
          a[start+3].data_ptr<bool>(),a[start+4].data_ptr<bool>(),a[start+5].data_ptr<int64_t>(),
          lookup.data_ptr<int64_t>(),lookup.size(0),lookup.size(1),p[0],p[1],p[2]};
}
}

void evidence_plan(const std::vector<at::Tensor>& a,const std::vector<at::Tensor>& o,
                   const std::vector<double>& p,const std::vector<int64_t>& m) {
  // Inputs: uv, projection integers/flags, query rows, xyz/normal/spacing,
  // valid/entity, observation pixels, lookup. Outputs are documented in Python.
  TORCH_CHECK(a.size()==11&&o.size()==7&&p.size()==6&&m.size()==7,"Evidence plan argument count");
  const int device=a[0].get_device();const c10::cuda::CUDAGuard guard(device);
  const at::ScalarType ad[]={at::kDouble,at::kLong,at::kBool,at::kLong,at::kDouble,at::kDouble,at::kDouble,at::kBool,at::kBool,at::kLong,at::kLong};
  const at::ScalarType od[]={at::kLong,at::kLong,at::kLong,at::kDouble,at::kBool,at::kLong,at::kLong};
  for(int i=0;i<11;++i)checked(a[i],ad[i],device);for(int i=0;i<7;++i)checked(o[i],od[i],device);
  const int64_t n=a[0].size(0);if(!n)return;
  TORCH_CHECK(a[0].numel()==2*n&&a[1].numel()==2*n&&a[2].numel()==8*n&&a[3].numel()==n,"Projection row alignment");
  const int64_t sizes[]={n,8*n,8*n,4*n,4*n,2*n,2*n};
  for(int i=0;i<7;++i)TORCH_CHECK(o[i].numel()==sizes[i],"Evidence output size");
  plan_kernel<<<(n+THREADS-1)/THREADS,THREADS,0,c10::cuda::getCurrentCUDAStream(device)>>>(n,
    a[0].data_ptr<double>(),a[1].data_ptr<int64_t>(),a[2].data_ptr<bool>(),a[3].data_ptr<int64_t>(),geometry(a,4,p),
    m[0],m[1],m[2],m[3],m[4],m[5],p[3],p[4],p[5],m[6],
    o[0].data_ptr<int64_t>(),o[1].data_ptr<int64_t>(),o[2].data_ptr<int64_t>(),o[3].data_ptr<double>(),
    o[4].data_ptr<bool>(),o[5].data_ptr<int64_t>(),o[6].data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void evidence_connect(const std::vector<at::Tensor>& a,const std::vector<at::Tensor>& o,
                      const std::vector<double>& p) {
  TORCH_CHECK(a.size()==14&&o.size()==3&&p.size()==3,"Evidence connectivity argument count");
  const int device=a[0].get_device();const c10::cuda::CUDAGuard guard(device);
  const at::ScalarType ad[]={at::kLong,at::kDouble,at::kDouble,at::kDouble,at::kBool,at::kBool,at::kLong,at::kLong,
                           at::kLong,at::kBool,at::kLong,at::kLong,at::kLong,at::kLong};
  for(int i=0;i<14;++i)checked(a[i],ad[i],device);
  checked(o[0],at::kInt,device);checked(o[1],at::kInt,device);checked(o[2],at::kBool,device);
  const int64_t count=a[10].numel();if(!count)return;
  TORCH_CHECK(count<=2147483647&&a[11].numel()==2*count&&a[12].numel()==2*count&&a[13].numel()==count+1,
              "Connectivity box alignment");
  connect_kernel<<<count,THREADS,0,c10::cuda::getCurrentCUDAStream(device)>>>(a[0].data_ptr<int64_t>(),geometry(a,1,p),
    a[8].data_ptr<int64_t>(),a[9].data_ptr<bool>(),a[10].data_ptr<int64_t>(),a[11].data_ptr<int64_t>(),
    a[12].data_ptr<int64_t>(),a[13].data_ptr<int64_t>(),o[0].data_ptr<int>(),o[1].data_ptr<int>(),o[2].data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void evidence_finish(const std::vector<at::Tensor>& a,const std::vector<at::Tensor>& o,
                     const std::vector<int64_t>& m) {
  TORCH_CHECK(a.size()==12&&o.size()==4&&m.size()==6,"Evidence finish argument count");
  const int device=a[0].get_device();const c10::cuda::CUDAGuard guard(device);
  const at::ScalarType ad[]={at::kLong,at::kBool,at::kLong,at::kLong,at::kLong,at::kLong,at::kDouble,at::kBool,
                           at::kDouble,at::kDouble,at::kLong,at::kLong};
  for(int i=0;i<12;++i)checked(a[i],ad[i],device);
  checked(o[0],at::kDouble,device);checked(o[1],at::kDouble,device);checked(o[2],at::kLong,device);checked(o[3],at::kBool,device);
  const int64_t n=a[2].numel();if(!n)return;
  for(const auto& item:o)TORCH_CHECK(item.numel()==n,"Evidence finish output rows");
  finish_kernel<<<(n+THREADS-1)/THREADS,THREADS,0,c10::cuda::getCurrentCUDAStream(device)>>>(n,
    a[0].data_ptr<int64_t>(),a[1].data_ptr<bool>(),a[2].data_ptr<int64_t>(),a[3].data_ptr<int64_t>(),
    a[4].data_ptr<int64_t>(),a[5].data_ptr<int64_t>(),a[6].data_ptr<double>(),a[7].data_ptr<bool>(),
    a[8].data_ptr<double>(),a[9].data_ptr<double>(),a[10].data_ptr<int64_t>(),a[11].data_ptr<int64_t>(),
    m[0],m[1],m[2],m[3],m[4],m[5],o[0].data_ptr<double>(),o[1].data_ptr<double>(),o[2].data_ptr<int64_t>(),o[3].data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
