module SVT_TASK1
using LinearAlgebra
using ForwardDiff
using Plots

function tridiag_algorithm(f::AbstractVector{T}) where T<:AbstractFloat
    n = length(f)
    a₊₁ = zeros(T, n - 1)
    a = zeros(T, n)
    a₋₁ = zeros(T, n - 1)
    b = copy(f)

    # Initialize the tridiagonal matrix coefficients
    for i in 1:n-1
        a₋₁[i] = T(-1.0)
        a[i] = T(2.0)
        a₊₁[i] = T(-1.0)
    end
    a[n] = T(2.0)

    # Forward substitution
    for i in 2:n
        m = a₋₁[i-1] / a[i-1]
        a[i] = a[i] - m * a₊₁[i-1]
        b[i] = b[i] - m * b[i-1]
    end

    # Backward substitution
    x = zeros(T, n)
    x[n] = b[n] / a[n]
    for i in n-1:-1:1
        x[i] = (b[i] - a₊₁[i] * x[i+1]) / a[i]
    end
    return x
end

function get_f(N::Int64)
    precise_solution(x) = sin(4x) * cos(3x)
    df(x) = ForwardDiff.derivative(precise_solution, x)
    ddf(x) = ForwardDiff.derivative(df, x)
    h = 1.0 / N
    f = zeros(Float64, N - 1)
    for i in 2:N
        xᵢ = (i - 1) * h
        f[i-1] = -ddf(xᵢ)
    end
    return f
end

function get_solution(N::Int64, a::T, b::T) where T<:AbstractFloat
    f = get_f(N)
    h = 1.0 / N
    f[1] += a / h^2
    f[end] += b / h^2
    u_inner = tridiag_algorithm(f)
    u_full = zeros(T, N + 1)
    u_full[1] = a
    u_full[2:end-1] = u_inner
    u_full[end] = b
    return u_full
end

function convergence_study(N_values::Vector{Int}, a::Float64, b::Float64)
    L2_errors = Float64[]
    C_errors = Float64[]
    hs = Float64[]

    for N in N_values
        h = 1.0 / N
        u_num = get_solution(N, a, b)
        x = range(0, 1, length=N + 1)
        u_exact = sin.(4x) .* cos.(3x)

        err = u_num .- u_exact
        L2 = sqrt(sum(err .^ 2) * h)
        C = maximum(abs.(err))

        push!(L2_errors, L2)
        push!(C_errors, C)
        push!(hs, h)
    end

    # Построение графиков
    plot(hs, L2_errors,
        xscale=:log10, yscale=:log10,
        marker=:circle, label="L₂ error",
        xlabel="h", ylabel="error",
        title="Convergence of finite difference scheme")
    plot!(hs, C_errors,
        marker=:square, label="C (max) error")

    display(plot!())
    savefig("convergence.png")
    return (L2_errors, C_errors, hs)
end

N_list = [10, 20, 40, 80, 160, 320]
a = 0.0
b = sin(4) * cos(3)
convergence_study(N_list, a, b)

end # module SVT_TASK1
