import PyPlot
plt = PyPlot

println("Done getting PyPlot")

bounded_thresh = 2
max_iter = 100
z0 = 0
resolution = 1000

x = collect(range(-1.5, 0.5, length=resolution))
y = collect(range(-1, 1, length=resolution))
Re = transpose(repeat(x, 1, resolution))
Im = repeat(y, 1, resolution) * im
Z = Re + Im

iters = Array{Int}(undef, (resolution, resolution))


function zn(z, c)
    return z^2 + c
end


function get_modulus(c)
    z = z0
    num_iter = 0
    for i in 1:max_iter
        num_iter = i
        z = zn(z, c)
        modulus_z = abs(z)
        if modulus_z > bounded_thresh
            break
        end
    end
    return num_iter
end


for i in 1:size(Z, 1)*size(Z,2)
    iters[i] = get_modulus(Z[i])
end

plt.imshow(iters)
plt.show()
