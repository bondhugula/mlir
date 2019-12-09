
This information goes together with this article on high-performance code
generation in MLIR:
https://github.com/bondhugula/mlir/blob/hop/g3doc/HighPerfCodeGen.md

### On running the DGEMM benchmark for experimentation

See the benchmark/ directory in mlir/.

To execute the included benchmark:

```
$ mlir-opt -hopt -hopt-vect -hopt-unroll -hopt-copy -hopt-scalrep benchmark/dgemm-tiled-benchmark.mlir  -convert-linalg-to-loops   -convert-linalg-to-llvm -convert-std-to-llvm  -canonicalize  | mlir-cpu-runner  -O3  -e main -time -reps=5   -entry-point-result=void    -shared-libs=lib/libmlir_runner_utils.so > /dev/null
```

Take a look at the generated MLIR by running this and adding/removing flags one
by one:

```
$ mlir-opt -hopt -hopt-vect -hopt-copy -hopt-unroll -hopt-scalrep benchmark/dgemm-tiled-benchmark.mlir
```

### Command-line flags

**-hopt**: Customized matmul optimization sequence (based on the BLIS schedule)
       where you can enable the following opts incrementally.

**-hopt-vect**: Enable auto-vectorization.

**-hopt-copy**: Enable packing of memrefs.

**-hopt-unroll**: Enable unroll-and-jam and unroll.

Any combination of these could be used. The only optimization step not included
here is of loop tiling: as such, we start from an already tiled loop nest in
dgemm-tiled-benchmark.mlir (albeit with no other optimizations on it).
Performing the tiling via the existing utilities mlir::tile and
mlir::interchange is left as an exercise to the reader. :)

**To try some of these optimizations standalone**:

**-affine-vectorize**: Auto-vectorization (completely different from the -affine-vectorize/"super vectorizer" in MLIR tree).

**-affine-scalrep**: scalar replacement

```
$ mlir-opt -affine-vectorize -affine-scalrep benchmark/dgemm-tiled-benchmark.mlir
```

Please raise an issue at https://github.com/bondhugula/mlir/ if you find
something unexpected.