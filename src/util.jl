# Return the `i`-th column of the array `A` as an `SVector`.
@inline function extract_svector(A, ::Val{NDIMS}, i) where {NDIMS}
    return SVector(ntuple(@inline(dim->A[dim, i]), NDIMS))
end

# When particles end up with coordinates so big that the cell coordinates
# exceed the range of Int, then `floor(Int, i)` will fail with an InexactError.
# In this case, we can just use typemax(Int), since we can assume that particles
# that far away will not interact with anything, anyway.
# This usually indicates an instability, but we don't want the simulation to crash,
# since adaptive time integration methods may detect the instability and reject the
# time step.
# If we threw an error here, we would prevent the time integration method from
# retrying with a smaller time step, and we would thus crash perfectly fine simulations.
@inline function floor_to_int(i)
    if isnan(i) || i > typemax(Int)
        return typemax(Int)
    elseif i < typemin(Int)
        return typemin(Int)
    end

    return floor(Int, i)
end

"""
    @threaded x for ... end

Run either a threaded CPU loop or launch a kernel on the GPU, depending on the type of `x`.
Semantically the same as `Threads.@threads` when iterating over a `AbstractUnitRange`
but without guarantee that the underlying implementation uses `Threads.@threads`
or works for more general `for` loops.

The first argument must either be a `KernelAbstractions.Backend` or an array from which the
backend can be derived to determine if the loop must be run threaded on the CPU
or launched as a kernel on the GPU. Passing `KernelAbstractions.CPU()` will run the GPU
kernel on the CPU.

In particular, the underlying threading capabilities might be provided by other packages
such as [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl).

!!! warn
    This macro does not necessarily work for general `for` loops. For example,
    it does not necessarily support general iterables such as `eachline(filename)`.
"""
macro threaded(system, expr)
    # Reverse-engineer the for loop.
    # `expr.args[1]` is the head of the for loop, like `i = eachindex(x)`.
    # So, `expr.args[1].args[2]` is the iterator `eachindex(x)`
    # and `expr.args[1].args[1]` is the loop variable `i`.
    iterator = expr.args[1].args[2]
    i = expr.args[1].args[1]
    inner_loop = expr.args[2]

    # Assemble the `for` loop again as a call to `parallel_foreach`, using `$i` to use the
    # same loop variable as used in the for loop.
    return esc(quote
                   PointNeighbors.parallel_foreach($iterator, $system) do $i
                       $inner_loop
                   end
               end)
end

# Use `Polyester.@batch` for low-overhead threading
@inline function parallel_foreach(f, iterator, x)
    Polyester.@batch for i in iterator
        @inline f(i)
    end
end

# On GPUs, execute `f` inside a GPU kernel with KernelAbstractions.jl
@inline function parallel_foreach(f, iterator,
                                  x::Union{AbstractGPUArray, KernelAbstractions.Backend})
    # On the GPU, we can only loop over `1:N`. Therefore, we loop over `1:length(iterator)`
    # and index with `iterator[eachindex(iterator)[i]]`.
    # Note that this only works with vector-like iterators that support arbitrary indexing.
    indices = eachindex(iterator)
    ndrange = length(indices)

    # Skip empty loops
    ndrange == 0 && return

    backend = KernelAbstractions.get_backend(x)

    # Call the generic kernel that is defined below, which only calls a function with
    # the global GPU index.
    generic_kernel(backend)(ndrange = ndrange) do i
        @inline f(iterator[indices[i]])
    end

    KernelAbstractions.synchronize(backend)
end

@kernel function generic_kernel(f)
    i = @index(Global)
    @inline f(i)
end
