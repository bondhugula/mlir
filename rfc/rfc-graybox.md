# RFC - affine.graybox op
https://github.com/bondhugula/mlir/blob/graybox/rfc/rfc-graybox.md

This proposal is on adding a new op named *affine.graybox* to MLIR's [affine dialect](https://github.com/tensorflow/mlir/blob/master/g3doc/Dialects/Affine.md).
The op allows the polyhedral form to be used without the need for outlining to
functions, and without the need to turn affine ops such as affine.for, affine.if
into standard unrestricted for's or to "list of basic blocks" control flow
respectively. In particular, with an *affine.graybox*, it is possible to represent
*every* load and store operation using an *affine.load* and *affine.store*
respectively.

Some form of this op was prototyped by @Alex Zinenko a while ago as discussed on
this thread:
https://groups.google.com/a/tensorflow.org/forum/#!topic/mlir/kxbrc6i59go
But the implementation is dated and there is no publicly available documentation
on it. This proposal has aspects that are different from the previous design.


## Op Description

1. The *affine.graybox* op has zero results, zero or more operands, and holds
   a single region, which is a list of zero or more blocks. The op's region can
   have zero or more arguments, each of which can only be a *memref*. The
   operands bind 1:1 to its region's arguments.  The op can't use any memrefs
   defined outside of it, but can use any other SSA values that dominate it. Its
   region's blocks can have terminators the same way as current MLIR functions
   (FuncOp) can. Control from any *return* ops in its region returns to right
   after the *affine.graybox* op.  The op will have the trait *FunctionLike*.

2. The requirement for an SSA value to be a valid symbol
   ([mlir::isValidSymbol](https://github.com/tensorflow/mlir/blob/3671cf5558a273a865007405503793746e4ddbb7/lib/Dialect/AffineOps/AffineOps.cpp#L128))
   changes so that it also includes (a) symbols at the top-level of any
   *affine.graybox*, and (b) those values that dominate an affine graybox. In
   the latter case, symbol validity is sensitive to the enclosing graybox. As
   such, there has to be an additional method: mlir::isValidSymbol(Value \*v,
   Operation \*op) to check for symbol validity for use in the specific op. See
   design alternatives towards the end.

3. An *affine.graybox* is not isolated from above. All SSA values (other than
   memrefs) that dominate the op can be used in the graybox. Constants and
   other replacements can freely propagate into it. Since memrefs from outside
   can't be used in the graybox and have to be explicitly provided as operands,
   canonicalization or replacement for them will only happen through rewrite
   patterns registered on *affine.graybox* op. More on this further below.

4. *affine.graybox* is a non-executing op that is eventually discarded by
   -lower-affine.

## Syntax

```mlir {.mlir}
// Custom form syntax.
affine.graybox `[` memref-use-list `]` `:` memref-type-list `{`
  block+
`}`
```

## Examples

Here are three examples: one related to non-affine control flow, one to
non-affine loop bounds, and the third to non-affine data accesses that can be
represented via affine.grayboxes without having to outline the function or
without having to use std load/stores: the latter two are the current
possibilities of representing them.

### Example 1

Here is how a simple typical non-affine control flow is represented with
an affine.graybox; note that all load/stores are in it are affine. This example
is used in the Rationale document to show how outlining to a separate function
allows representation using affine constructs:
https://github.com/tensorflow/mlir/blob/master/g3doc/Rationale.md#Examples

```
// A simple linear search in every row of a matrix.
for (i = 0; i < N; i++) {
  for (j = 0; j < N; j++) {
    // Dynamic/non-affine control flow
    if (a[i][j] == key) {
      s[i] = j;
      break;
    }
  }
}
```

```mlir {.mlir}
func @search(%A : memref<?x?xi32, %S : <?xi32>, %key : i32) {
  %ni = dim %A, 0 : memref<?x?xi32>
  // This loop can be parallelized.
  affine.for %i = 0 to %ni {
    affine.graybox [%A, %S] : (memref<?x?xi32>, memref<?xi32>) {
      %nj = dim %A, 1 : memref<?x?xi32>
      br ^bb1(%c0)

    ^bb1(%j: i32)
      %p1 = cmpi "lt", %j, %nj : i32
      cond_br %p1, ^bb2, ^bb5

    ^bb2:
      %v = affine.load %A[%i, %j] : memref<?x?xi32>
      %p2 = cmpi "eq", %v, %key : i32
      cond_br %p2, ^bb3(%j), ^bb4

    ^bb3(%j: i32)
      affine.store %j, %S[%i] : memref<?xi32>
      br ^bb5

    ^bb4:
      %jinc = addi %j, %c1 : i32
      br ^bb1(%jinc)

    ^bb5:
      return
    }
  }
  return
}
```

### Example 2

Non-affine loop bounds:

```
for (i = 0; i < N; i++)
  for (j = 0; j < N; j++)
    // non-affine loop bound for k loop
    for (k=0; k<pow(2,j); k++)
       for (l=0; l<N; l++) {
        // block loop body
        ...

```


```mlir {.mlir}
func @nest(%n : i32) {
  %c2 = constant 2 : index
  affine.for %i = 0 to %n {
    affine.for %j = 0 to %n {
      affine.graybox [] {
        %pow = call @powi(%c2, %j) : (index, index) ->  index
        affine.for %k = 0 to %pow {
          affine.for %l = 0 to %n {
            ...
          }
        }
        return
      }  // graybox end
    }
  }
  return
}
```

### Example 3

Non-affine loop bounds.

```
for (i = 0; i < N; ++i) {
  A[B[i]]++;
}
```

```mlir {.mlir}
func @non_affine_load_store(%A : memref<100xf32>, %B : memref<100xf32>) {
  %cf1 = constant 1.0 : f32
  for %i = 0 to 100 {
    %v = affine.load %B[%i] : memref<100xf32>
    affine.graybox [] {
      // %v is now a symbol here.
      %s = affine.load %A[%v] : memref<100xf32>
      %o = addf %s, %cf1 : f32
      affine.store %o, %A[%v] : memref<100xf32>
      return;
    }
  }
  return
}
```


## Utilities and Passes

* **Hoist or eliminate affine grayboxes**

   There will be a function pass that will hoist or eliminate unnecessary
   *affine.graybox* ops, i.e., when an *affine.graybox* can be eliminated or hoisted
   without violating the dimension and symbol requirements. The propagation of
   constants and other simplification that happens in scalar optimizations /
   canonicalization helps get rid of affine.grayboxes. As such it's useful to
   have non-memref SSA values be implicitly captured and not isolate them.

* **Walkers**

   There has to be a new walkAffine method that will walk everything except
   regions of an affine.graybox. Most polyhedral/affine passes will see
   *affine.graybox* as opaque *for any walks from above*.

   An affine pass's runOnFunction should be changed to to run on that function
   as well as every *affine.graybox* op in it. Unfortunately, they have to be done
   sequentially only because the "declaration" of the *affine.graybox* and the
   "imperative" call to it are one thing - the affine grayboxes could have
   otherwise been processed in parallel. In summary, there can be an
   AffineFunctionPass that needs to only implement an runOnOp(op) where op is
   either a FuncOp or an AffineGrayBoxOp.

   Some of the current polyhedral passes/utilities can continue using walk (for
   eg. [normalizeMemRefs](https://github.com/tensorflow/mlir/blob/331c663bd2735699267abcc850897aeaea8433eb/include/mlir/Transforms/Utils.h#L89), while many will just have to be changed to use walkAffine.

* **Simplification / Canonicalization**

   There has to be a simplification that drops unused block arguments from
   regions of ops that aren't function ops (since this is easy for non function
   ops) - in case this isn't already in place. This will allow removal of dead
   memrefs that could otherwise be blocked by operand uses in affine.graybox ops
   with the corresponding region arguments not really having any uses inside.
   Given this, no additional bookkeeping is needed as a result of having memrefs
   as explicit operands for gray boxes. [MemRefCastFold](https://github.com/tensorflow/mlir/blob/ef77ad99a621985aeca1df94168efc9489de95b6/lib/Dialect/StandardOps/Ops.cpp#L228) is the only canonicalization pattern
   that the *affine.graybox* has to implement, and this is easily/cleanly done
   (by replacing the argument and its uses with a memref of a different type).
   Overall, having memrefs as explicit arguments is a good middle ground to
   make it easier to let standard SSA passes / scalar optimizations /
   canonicalizations work unhindered in conjunction with polyhedral passes, and
   with the latter not worrying about explicitly checking for escaping/hidden
   memref accesses. More discussion a little below in the next section.


*  There are situations/utilities where one can consistently perform
   rewriting/transformation/analysis cutting across grayboxes. One example is
   [normalizeMemRefs](https://github.com/tensorflow/mlir/blob/331c663bd2735699267abcc850897aeaea8433eb/include/mlir/Transforms/Utils.h#L89), which turns all non-identity layout maps into identity
   ones. Having memrefs explicitly captured is a hindrance here, but
   mlir::replaceAllMemrefUsesWith can be extended to transparently perform the
   replacement inside any affine grayboxes encountered if the caller says so.
   In other cases like scalar replacement, memref packing / explicit copying,
   DMA generation, pipelining of DMAs, transformations are supposed to be
   blocked by those boundaries because the accesses inside the graybox can't be
   meaningfully analyzed in the context of the surrounding code. As such, the
   memrefs there are treated as escaping / non-dereferencing.

*  In the presence of affine constructs, the inliner can now simply inline
   functions by putting the callee inside an affine graybox, without having to
   worry about symbol restrictions.

* There has to be a mlir::getEnclosingAffineGrayBox(op) that returns the closest
  enclosing *affine.graybox* op or null if it hits a function op.


## Other Benefits and Implications

1. The introduction of this op allows arbitrary control flow (list of basic
   blocks with terminators) to be used within and mixed with affine.fors/ifs
   while staying in the same function. Such a list of blocks will be carried by
   an *affine.graybox* op whenever it's not at the top level.

2. Non-affine data accesses can now be represented through
   *affine.load/affine.store* without the need for outlining.

3. Symbol restrictions for affine constructs will no longer restrict inlining:
   any function can now be inlined into another by enclosing the just inlined
   function into a graybox.

4. Any access to a memref can be represented with an *affine.load/store*. This is
   quite useful in order to reuse existing passes more widely (for eg. to
   perform scalar replacement on affine accesses) --- there is no reason
   memref-dataflow-opt won't work on / shouldn't be reused on straightline code,
   which is always a valid *affine.graybox* region (at the limit, the innermost
   loop body is a valid *affine.graybox* under all circumstances.

5. Any countable C-style 'for' without a break/continue can be represented as an
   affine.for (irrespective of bounds). Any if/else without a continue/break
   inside can be represented as an affine.if. The rest are just represented as a
   list of basic blocks with an enclosing *affine.graybox* if not at the
   top-level.

6. SSA scalar opts work across ``affine boundaries'' without having to be
   interprocedural.

## Rationale and Design Alternatives - What to Capture as Arguments?

An alternative design is to allow all SSA values including memrefs to be
implicitly captured, i.e., zero operands and arguments for the op. This is
however inconvenient for all polyhedral transformations and analyses which will
have to check and scan any affine.grayboxes encountered to see if any memrefs
are being used therein, and if so, they would most likely treat them as if the
memrefs were being passed to a function call. This would be the case with
dependence analysis, memref region computation/analysis, fusion, explicit
copying/packing, DMA generation, pipelining, scalar replacement and anything
depending on the former analyses (like tiling, unroll and jam). Having memrefs
as explicit operands/arguments is a good middle ground to make it easier to let
standard SSA passes / scalar optimization / canonicalization work
unhindered in conjunction with polyhedral passes, and with the latter not
worrying about  explicitly checking for escaping/hidden memref accesses.

Furthermore, a memref anyway never folds to a constant. The only
canonicalization related to a memref currently is a [memref_cast
fold](https://github.com/tensorflow/mlir/blob/ef77ad99a621985aeca1df94168efc9489de95b6/lib/Dialect/StandardOps/Ops.cpp#L228)
which can easily be extended to fold  with an *affine.graybox* op (update its
argument and all uses inside).  As such, there aren't any cases where the
argument list has to be shrunk/grown from the outside. And for the cases where
the types have to be updated, it's straightforward since there is sort of only a
single use for that op instance (it's not declarative or "callable" from
elsewhere like a FuncOp).

Another design point could be of requiring symbols associated with the
affine constructs used in a graybox, but defined outside, to be explicitly
listed as operands/arguments, in addition to the memrefs used. This makes
isValidSymbol really simple. One won't need isValidSymbol(Value \*v,
AffineGrayBoxOp op). Anything that is at the top-level of an *affine.graybox* op
or its region argument will become a valid symbol. However, other than this, it
doesn't simplify anything else. Instead, it adds/duplicates a
bookkeeping with respect to propagation of constants, similar, to some
extent, to the argument rewriting done for interprocedural constant propagation.
Similarly, the other extreme of requiring everything from the outside used in an
*affine.graybox* to be explicitly listed as its operands and region arguments is
even worse on this front.

In summary, it appears that the requirement to explicitly capture only the
memrefs inside an affine.graybox's region is a good middle ground and better
than other options.


--
You received this message because you are subscribed to the Google Groups "MLIR" group.
To unsubscribe from this group and stop receiving emails from it, send an email to mlir+unsubscribe@tensorflow.org<mailto:mlir+unsubscribe@tensorflow.org>.
To view this discussion on the web visit https://groups.google.com/a/tensorflow.org/d/msgid/mlir/b229eeaf-2a53-4df3-a690-2ef7f0946232%40tensorflow.org<https://groups.google.com/a/tensorflow.org/d/msgid/mlir/b229eeaf-2a53-4df3-a690-2ef7f0946232%40tensorflow.org?utm_medium=email&utm_source=footer>.

