// RUN: mlir-opt %s -affine-vectorize | FileCheck %s

// CHECK-LABEL: func @loop1d
func @loop1d(%A: memref<2048x2048xf32>) {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      %v = affine.load %A[%i, %j] : memref<2048x2048xf32>
      affine.store %v, %A[%i, %j] : memref<2048x2048xf32>
    }
  }
  return
}
// CHECK-NEXT: %0 = memref_shape_cast %arg0 : memref<2048x2048xf32> to memref<2048x256xvector<8xf32>>
// CHECK-NEXT: affine.for %arg1 = 0 to 1024 {
// CHECK-NEXT:   affine.for %arg2 = 0 to 128 {
// CHECK-NEXT:     %1 = affine.load %0[%arg1, %arg2] : memref<2048x256xvector<8xf32>>
// CHECK-NEXT:     affine.store %1, %0[%arg1, %arg2] : memref<2048x256xvector<8xf32>>
// CHECK-NEXT:   }
// CHECK-NEXT: }


// CHECK-LABEL: func @matmul_ijk
func @matmul_ijk(%A: memref<2048x2048xf64>, %B: memref<2048x2048xf64>, %C: memref<2048x2048xf64>) {
  affine.for %arg3 = 0 to 2048 {
    affine.for %arg4 = 0 to 2048 {
      affine.for %arg5 = 0 to 2048 {
        %a = affine.load %A[%arg3, %arg5] : memref<2048x2048xf64>
        %b = affine.load %B[%arg5, %arg4] : memref<2048x2048xf64>
        %ci = affine.load %C[%arg3, %arg4] : memref<2048x2048xf64>
        %p = mulf %a, %b : f64
        %co = addf %ci, %p : f64
        affine.store %co, %C[%arg3, %arg4] : memref<2048x2048xf64>
      }
    }
  }
  return
// CHECK:       %0 = memref_shape_cast %arg2 : memref<2048x2048xf64> to memref<2048x512xvector<4xf64>>
// CHECK-NEXT:  %1 = memref_shape_cast %arg1 : memref<2048x2048xf64> to memref<2048x512xvector<4xf64>>
// CHECK-NEXT:  affine.for %arg3 = 0 to 2048 {
// CHECK-NEXT:    affine.for %arg4 = 0 to 512 {
// CHECK-NEXT:      affine.for %arg5 = 0 to 2048 {
// CHECK-NEXT:        %2 = affine.load %arg0[%arg3, %arg5] : memref<2048x2048xf64>
// CHECK-NEXT:        %3 = splat %2 : vector<4xf64>
// CHECK-NEXT:        %4 = affine.load %1[%arg5, %arg4] : memref<2048x512xvector<4xf64>>
// CHECK-NEXT:        %5 = affine.load %0[%arg3, %arg4] : memref<2048x512xvector<4xf64>>
// CHECK-NEXT:        %6 = mulf %3, %4 : vector<4xf64>
// CHECK-NEXT:        %7 = addf %5, %6 : vector<4xf64>
// CHECK-NEXT:        affine.store %7, %0[%arg3, %arg4] : memref<2048x512xvector<4xf64>>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
}

// Outer loop imperfect nest vectorization here.

#map7 = (d0) -> (d0 * 16)
#map8 = (d0) -> (d0 * 16 + 16)
#map9 = (d0) -> (d0 * 8)
#map10 = (d0) -> (d0 * 8 + 8)

// CHECK-LABEL: func @sgemm_blis_tiled
func @sgemm_blis_tiled(%A: memref<2048x2048xf32>, %B: memref<2048x2048xf32>, %C: memref<2048x2048xf32>) {
  %c0 = constant 0 : index
  affine.for %arg3 = 0 to 16 {
    affine.for %arg4 = 0 to 4 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = #map9(%arg3) to #map10(%arg3) {
          affine.for %arg7 = #map7(%arg5) to #map8(%arg5) {
            affine.for %kk = 0 to 512 {
              affine.for %jj = 0 to 16 {
                %0 = affine.load %B[%arg4 * 512 + %kk, %arg6 * 16 + %jj] : memref<2048x2048xf32>
                affine.for %ii = 0 to 4 {
                  %1 = affine.load %A[%arg7 * 4 + %ii, %arg4 * 512 + %kk] : memref<2048x2048xf32>
                  %2 = affine.load %C[%arg7 * 4 + %ii, %arg6 * 16 + %jj] : memref<2048x2048xf32>
                  %3 = mulf %1, %0 : f32
                  %4 = addf %2, %3 : f32
                  affine.store %4, %C[%arg7 * 4 + %ii, %arg6 * 16 + %jj] : memref<2048x2048xf32>
                }
              }
            }
// CHECK:      affine.for %arg8 = 0 to 512 {
// CHECK-NEXT:   affine.for %arg9 = 0 to 2 {
// CHECK-NEXT:     %2 = affine.load %1[%arg4 * 512 + %arg8, %arg6 * 2 + %arg9] : memref<2048x256xvector<8xf32>>
// CHECK-NEXT:     affine.for %arg10 = 0 to 4 {
// CHECK-NEXT:       %3 = affine.load %arg0[%arg7 * 4 + %arg10, %arg4 * 512 + %arg8] : memref<2048x2048xf32>
// CHECK-NEXT:       %4 = splat %3 : vector<8xf32>
// CHECK-NEXT:       %5 = affine.load %0[%arg7 * 4 + %arg10, %arg6 * 2 + %arg9] : memref<2048x256xvector<8xf32>>
// CHECK-NEXT:       %6 = mulf %4, %2 : vector<8xf32>
// CHECK-NEXT:       %7 = addf %5, %6 : vector<8xf32>
// CHECK-NEXT:       affine.store %7, %0[%arg7 * 4 + %arg10, %arg6 * 2 + %arg9] : memref<2048x256xvector<8xf32>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
          }
        }
      }
    }
  }
  return
}