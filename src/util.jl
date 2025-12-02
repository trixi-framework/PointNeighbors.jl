# Return the `i`-th column of the array `A` as an `SVector`.
@inline function extract_svector(A, ::Val{NDIMS}, i) where {NDIMS}
    # Explicit bounds check, which can be removed by calling this function with `@inbounds`
    @boundscheck checkbounds(A, NDIMS, i)

    # Assume inbounds access now
    return SVector(ntuple(@inline(dim->@inbounds A[dim, i]), NDIMS))
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
    # `Base.floor(Int, i)` is defined as `trunc(Int, round(x, RoundDown))`
    rounded = round(i, RoundDown)

    # `Base.trunc(Int, x)` throws an `InexactError` in these cases, and otherwise
    # returns `unsafe_trunc(Int, rounded)`.
    if isnan(rounded) || rounded >= typemax(Int)
        return typemax(Int)
    elseif rounded <= typemin(Int)
        return typemin(Int)
    end

    # After making sure that `rounded` is in the range of `Int`,
    # we can safely call `unsafe_trunc`.
    return unsafe_trunc(Int, rounded)
end

abstract type AbstractThreadingBackend end

"""
    SerialBackend()

Pass as first argument to the [`@threaded`](@ref) macro to run the loop serially.
"""
struct SerialBackend <: AbstractThreadingBackend end

"""
    PolyesterBackend()

Pass as first argument to the [`@threaded`](@ref) macro to make the loop multithreaded
with `Polyester.@batch`.
"""
struct PolyesterBackend <: AbstractThreadingBackend end

"""
    ThreadsDynamicBackend()

Pass as first argument to the [`@threaded`](@ref) macro to make the loop multithreaded
with `Threads.@threads :dynamic`.
"""
struct ThreadsDynamicBackend <: AbstractThreadingBackend end

"""
    ThreadsStaticBackend()


Pass as first argument to the [`@threaded`](@ref) macro to make the loop multithreaded
with `Threads.@threads :static`.
"""
struct ThreadsStaticBackend <: AbstractThreadingBackend end

const ParallelizationBackend = Union{AbstractThreadingBackend, KernelAbstractions.Backend}

"""
    default_backend(x)

Select the recommended backend for an array `x`.
This allows to write generic code that works for both CPU and GPU arrays.

The default backend for CPU arrays is currently `PolyesterBackend()`.
For GPU arrays, the respective `KernelAbstractions.Backend` is returned.
"""
@inline default_backend(::AbstractArray) = PolyesterBackend()
@inline default_backend(x::AbstractGPUArray) = KernelAbstractions.get_backend(x)
@inline default_backend(x::PermutedDimsArray) = default_backend(x.parent)

"""
    @threaded backend for ... end

Run either a threaded CPU loop or launch a kernel on the GPU, depending on the `backend`.
Semantically the same as `Threads.@threads` when iterating over a `AbstractUnitRange`
but without guarantee that the underlying implementation uses `Threads.@threads`
or works for more general `for` loops.

Possible parallelization backends are:
- [`SerialBackend`](@ref) to disable multithreading
- [`PolyesterBackend`](@ref) to use `Polyester.@batch`
- [`ThreadsDynamicBackend`](@ref) to use `Threads.@threads :dynamic`
- [`ThreadsStaticBackend`](@ref) to use `Threads.@threads :static`
- Any `KernelAbstractions.Backend` to execute the loop as a GPU kernel

Use `default_backend(x)` to select the recommended backend for an array `x`.
This allows to write generic code that works for both CPU and GPU arrays.

!!! warning "Warning"
    This macro does not necessarily work for general `for` loops. For example,
    it does not necessarily support general iterables such as `eachline(filename)`.
"""
macro threaded(backend, expr)
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
                   PointNeighbors.parallel_foreach($iterator, $backend) do $i
                       $inner_loop
                   end
               end)
end

# Serial loop
@inline function parallel_foreach(f, iterator, ::SerialBackend)
    for i in iterator
        @inline f(i)
    end
end

# Use `Polyester.@batch`
@inline function parallel_foreach(f, iterator, ::PolyesterBackend)
    Polyester.@batch for i in iterator
        @inline f(i)
    end
end

# Use `Threads.@threads :dynamic`
@inline function parallel_foreach(f, iterator, ::ThreadsDynamicBackend)
    Threads.@threads :dynamic for i in iterator
        @inline f(i)
    end
end

# Use `Threads.@threads :static`
@inline function parallel_foreach(f, iterator, ::ThreadsStaticBackend)
    Threads.@threads :static for i in iterator
        @inline f(i)
    end
end

# On GPUs, execute `f` inside a GPU kernel with KernelAbstractions.jl
@inline function parallel_foreach(f, iterator, backend::KernelAbstractions.Backend)
    # On the GPU, we can only loop over `1:N`. Therefore, we loop over `1:length(iterator)`
    # and index with `iterator[eachindex(iterator)[i]]`.
    # Note that this only works with vector-like iterators that support arbitrary indexing.
    indices = eachindex(iterator)
    ndrange = length(indices)

    # Skip empty loops
    ndrange == 0 && return

    # Call the generic kernel that is defined below, which only calls a function with
    # the global GPU index.
    generic_kernel(backend)(ndrange = ndrange) do i
        @inbounds @inline f(iterator[indices[i]])
    end

    KernelAbstractions.synchronize(backend)
end

@kernel function generic_kernel(f)
    i = @index(Global)
    @inline f(i)
end
