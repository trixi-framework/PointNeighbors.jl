using Documenter
using Literate: Literate

# Get PointNeighbors.jl root directory
trixibase_root_dir = dirname(@__DIR__)

# Fix for https://github.com/trixi-framework/Trixi.jl/issues/668
if (get(ENV, "CI", nothing) != "true") &&
   (get(ENV, "TRIXIBASE_DOC_DEFAULT_ENVIRONMENT", nothing) != "true")
    push!(LOAD_PATH, trixibase_root_dir)
end

using PointNeighbors

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(PointNeighbors, :DocTestSetup,
                    :(using PointNeighbors); recursive = true)

# Copy files to not need to synchronize them manually
function copy_file(filename, replaces...; new_filename = lowercase(filename))
    content = read(joinpath(trixibase_root_dir, filename), String)
    content = replace(content, replaces...)

    header = """
    ```@meta
    EditURL = "https://github.com/trixi-framework/PointNeighbors.jl/blob/main/$filename"
    ```
    """
    content = header * content

    write(joinpath(@__DIR__, "src", new_filename), content)
end

copy_file("README.md", new_filename = "index.md")
copy_file("AUTHORS.md",
          "in the [LICENSE.md](LICENSE.md) file" => "under [License](@ref)")
# Add section `# License` and add `>` in each line to add a quote
copy_file("LICENSE.md",
          "[AUTHORS.md](AUTHORS.md)" => "[Authors](@ref)",
          "\n" => "\n> ", r"^" => "# License\n\n> ")

mkpath(joinpath(@__DIR__, "src", "tutorials"))

Literate.markdown(joinpath(@__DIR__, "literate", "src", "tut_basic_usage.jl"),
                  joinpath(@__DIR__, "src", "tutorials"))
Literate.markdown(joinpath(@__DIR__, "literate", "src", "tut_n_body.jl"),
                  joinpath(@__DIR__, "src", "tutorials"))
Literate.markdown(joinpath(@__DIR__, "literate", "src", "tut_periodicity.jl"),
                  joinpath(@__DIR__, "src", "tutorials"))
Literate.markdown(joinpath(@__DIR__, "literate", "src", "tut_gpu_usage.jl"),
                  joinpath(@__DIR__, "src", "tutorials"))
Literate.markdown(joinpath(@__DIR__, "literate", "src", "tut_advanced_usage.jl"),
                  joinpath(@__DIR__, "src", "tutorials"))

# Make documentation
makedocs(modules = [PointNeighbors],
         sitename = "PointNeighbors.jl",
         # Provide additional formatting options
         format = Documenter.HTML(
                                  # Disable pretty URLs during manual testing
                                  prettyurls = get(ENV, "CI", nothing) == "true",
                                  # Set canonical URL to GitHub pages URL
                                  canonical = "https://trixi-framework.github.io/PointNeighbors.jl/stable",
                                  # Set edit_link explicitly to avoid `git remote show origin` lookups.
                                  edit_link = "main"),
         # Explicitly specify documentation structure
         pages = [
             "Home" => "index.md",
             "Tutorials" => [
                 "Basic Usage" => joinpath("tutorials", "tut_basic_usage.md"),
                 "N-Body" => joinpath("tutorials", "tut_n_body.md"),
                 "Periodicity" => joinpath("tutorials", "tut_periodicity.md"),
                 "GPU Usage" => joinpath("tutorials", "tut_gpu_usage.md"),
                 "Advanced Usage" => joinpath("tutorials", "tut_advanced_usage.md")
             ],
             "API reference" => "reference.md",
             "Authors" => "authors.md",
             "License" => "license.md"
         ])

deploydocs(;
           repo = "github.com/trixi-framework/PointNeighbors.jl",
           devbranch = "main",
           push_preview = true)
