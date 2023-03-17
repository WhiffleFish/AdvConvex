module AdvConvex

using Reexport

include(joinpath("HW3","HW3.jl"))
@reexport using .HW3

include(joinpath("HW4","HW4.jl"))
@reexport using .HW4

end # module AdvConvex
