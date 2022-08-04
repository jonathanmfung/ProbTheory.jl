using Random
using Distributions

using Plots


u = Uniform(-1e5, 1e5) # floatmax(Float32)

sample_avg(xs) = mean(xs)
cummean(A) = cumsum(A) ./ (1:length(A))

const MEDIA_DIR = joinpath(@__DIR__, "media")

"    https://en.wikipedia.org/wiki/Law_of_large_numbers#Forms"
function LLN(;n = 600, file = "LLN.gif")
    Random.seed!(1234)

    v = cummean(rand(u,n))
    plt = plot(1, xlims=(0,n), title = "LLN",
                  xlab = "n", ylab = "sample mean",
                  dpi = 600)
    hline!(plt, [0], color = "red", label = "population mean")

    anim = @animate for i ∈ 1:n
        push!(plt, v[i])
    end
    gif(anim,
        joinpath(MEDIA_DIR, file),
        fps = 60)
end

"    https://en.wikipedia.org/wiki/Central_limit_theorem#Classical_CLT"
function CLT(;n = 600, file = "CLT.gif")
    Random.seed!(1234)

    v = Vector{Float64}()
    anim = @animate for i ∈ 1:n
        push!(v, sample_avg(rand(u,100)))
        histogram(v, xlims=(-3e4,3e4), ylims=(0,100),
                  bins = 20, legend = false, title = "CLT",
                  xlab = "sample mean", ylab = "count",
                  dpi = 600)
    end
    gif(anim,
        joinpath(MEDIA_DIR, file),
        fps = 60)
end

"    https://ubt.opus.hbz-nrw.de/opus45-ubtr/frontdoor/deliver/index/docId/732/file/Dissertation_Schulz.pdf
    https://math.stackexchange.com/questions/3335024/enhanced-berry-esseen-theorem-for-the-digits-of-sqrt2"
function BE_binom(n::Int64, p::Float64)
    q = 1 - p
    xs = 0:100.0

    bcdf = Distributions.binomcdf.(n, p, xs) .|> BigFloat
    ncdf = Distributions.normcdf.((xs .- n * p)/sqrt(n * p * q)) .|> BigFloat

    ksdist = maximum(abs.(bcdf - ncdf))
    bound = (sqrt(10) + 3)/(6 * sqrt(2*π)) * (p^2 + q^2)/sqrt(n * p * q)
    tightbound = (3 + abs(p - q))/(6 * sqrt(2*π) * sqrt(n * p * q))

    return [ksdist, bound, tightbound]
end

function BE_binom(t::Tuple{Int64, Float64})
    BE_binom(t[1], t[2])
end

function BE_binom_heatmap(; n = 500,
                          pstart = 0.001, pend = 0.999, len = 999,
                          file = "BE_binom_heatmap.png")
    ps = range(pstart, pend, length = len)
    vals = BE_binom.(Base.product(1:n, ps))

    difs = getindex.(vals, 3) - getindex.(vals, 1)

    plt = heatmap(difs.^(1/16),
                  xlab = "p * 1000", ylab = "n", colorbar_title = "tightbound - ksdist",
                  xtick = 0:200:1000,
                  ytick = 0:100:n,
                  c = :Blues,
                  dpi = 600)

    # heatmap(log.(difs), c = :Blues)
    # surface(difs.^(1/16), c = :Blues, camera = (210, 30))

    name, ext = splitext(file)
    file_ = string(name, "_", n, ext)

    savefig(plt,
            joinpath(MEDIA_DIR, file_))
end

function BE_binom_slice(; n = 20, len = 999, file = "BE_binom_slice.png")
    ps = range(0.001, 0.999, length = len)
    vals = BE_binom.(Base.product(1:n, ps))

    plt = scatter3d(xlims = (0,n), ylims = (0,1),
                    xlab = "n", ylab = "p", zlab = "tightbound - ksdist",
                    camera = (70, 60), legend = false, dpi = 600)

    for i in 1:n
        plt = plot!(plt,
                    fill(i,len),
                    ps,
                    (getindex.((vals[i,:]), 3) - getindex.((vals[i,:]), 1)).^(1/16),
                    # :winter, :seaborn_icefire_gradient, :deepsea
                    color = cgrad(:winter, n, categorical = true)[i]
                    )
    end
    savefig(plt,
            joinpath(MEDIA_DIR, file))
end

function BE_binom_error(n::Int64, p::Float64; file = "BE_binom_error.png")
    ts = hcat(BE_binom.(1:n, p)...)

    delta = ts[3,:] - ts[1,:]
    plt = plot(ts[1,:], label = "Δₜ", dpi = 600)
    plt = hline!(plt, [0], label = "0")
    savefig(plt,
            joinpath(MEDIA_DIR, file))
end

function BE_binom_bounds(n :: Int64)
    function validate(n::Int64, pstart::Float64, pend::Float64)
        vals = BE_binom.(Base.product(1:n, pstart:0.01:pend))
        diffs = getindex.(vals, 3) - getindex.(vals, 1)
        pred = sum((getindex.(vals, 3) - getindex.(vals, 1)) .< 0)
        println("For n = $n on p = [$pstart, $pend], there are $pred negative differences.")
    end

    validate(n, 0.001, 0.999)
    validate(n, 0.333, 0.666)
end
