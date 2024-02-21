using Distributions
using Random
using StatsBase
using NPZ
#using BenchmarkTools
#using PyPlot
# From Basic_ranked_diff_20.jl

function update_row!(matrix, row_ind, new_row)
    """This function updates a row in a matrix (matrix) 
    with a new row (new_row) at index (row_ind). 
    It changes the input matrix in-place.
    """    
    for i in 1:length(matrix[1,:])
        matrix[row_ind, i] = new_row[i]
    end
end

function kateri_bin(bini, stevilo)#"bini" je seznam z robovi binov, "stevilo" zelimo razporediti v bin
    """This function tells you in which bin given number belongs if you know the coordinates of the edges of bins.
    bini (List): List of bin edges.
    stevilo (Float): number that we want to put into bins.

    Returns:
        Int: Index of a bin in which number (stevilo) belongs.
    """    
    #stevilo robov
    r = length(bini)
    #stevilo binov
    b = r + 1
    # sirina bina
    w = (last(bini)-bini[1])/(r-1)
    if stevilo<bini[1]
        index = 1
    elseif stevilo>last(bini)
        index = b
    else
        index = trunc(Int64,((stevilo-bini[1])÷w)+2)
    end
    return index
end
    
function ranked_diffusion2p_st_BETTER_MORE(N, c, T, time_steps, l)
    """Same as 'ranked_diffusion2' but improved implementation. 
    Initial condition is a step of width l.

    Returns:
        List: positions of particles at the last time.
    """  
    dx = 2*l / (N-1)
    positions = [-l+dx*i for i in 0:(N-1)]
    vsotice = zeros(Float64, N)
    for t in 1:time_steps
        # najprej sortiram arej, ampak tako da sortiram indekse. (sortiran_i je seznam ki ima indekse prvotnega areja, v sortiranem vrstnem redu)
        sortiran_i = sortperm(positions)
        fill!(vsotice, 0.0)
        for i in 1:N
            vsotice[sortiran_i[i]] += N + 1 - 2*i
        end
        positions .+= -c.*vsotice .+ (sqrt(2*T).*randn(N))
    end
    return positions
end

# Z enim hitrim testom sem preverla da kao dela
function ranked_diffusion2p_delta_BETTER_MORE(N, c, T, time_steps)
    """Same as ranked_diffusion2 but better implemented,
     initial condition is a delta function.

    Returns:
        List: Array of position of particles at the last time.
    """ 
    positions = zeros(Float64, N)
    vsotice = zeros(Float64, N)
    for t in 1:time_steps
        # najprej sortiram arej, ampak tako da sortiram indekse. (sortiran_i je seznam ki ima indekse prvotnega areja, v sortiranem vrstnem redu)
        sortiran_i = sortperm(positions)
        fill!(vsotice, 0.0)
        for i in 1:N
            vsotice[sortiran_i[i]] += N + 1 - 2*i
        end
        positions .+= -c.*vsotice .+ (sqrt(2*T).*randn(N))
    end
    return positions
end

function averaged_density2(N, c, T, time_steps, averaging, st_binov, sirina, l)
    """A function using 'ranked_diffusion2p_st_BETTER' (step initial c.). 
    And producing a distribution of final positions by running the ranked diffusion process many times.
    averaging (int): number of times to run the ranked diffusion (number of samples).
    st_binov (Int): number of bins for constructing final histogram of density.
    sirina (Float): width of the histogram.
    l (Float): width of the initial condition.

    Returns:
        List: Histogram showing averaged density at the final time.
    """ 
    bini = LinRange(-sirina, sirina, st_binov-1)
    histogram = zeros(Float64,st_binov)
    for i in 1:averaging
        if l == 0
            a = ranked_diffusion2p_delta_BETTER_MORE(N, c, T, time_steps)
            for k in 1:N
                ind = kateri_bin(bini, a[k])
                histogram[ind] += 1
            end
        else
            a = ranked_diffusion2p_st_BETTER_MORE(N, c, T, time_steps, l)
            for k in 1:N
                ind = kateri_bin(bini, a[k])
                histogram[ind] += 1
            end
        end
    end
    if l==0
        npzwrite("./Data/Density_delta_N$(N)_c$(c)_t$(time_steps)_avrg$(averaging)_l$(l).npz", histogram./averaging)
    else
        npzwrite("./Data/Density_step_N$(N)_c$(c)_t$(time_steps)_avrg$(averaging)_l$(l).npz", histogram./averaging)
    end
    return 1
end

#const c = 0.01
const T = 1
#const N = 100
#const time_steps = 100
const l = 0
const averaging = 1000
const st_bin = 100

a = averaged_density2(100, 0.01, T, 10, averaging, st_bin, 4*10, l)
println("The simulation for c = 0.01, T=$T, N=100, time steps = 10, l=$(l) is finished.")

a = averaged_density2(100, 0.01, T, 100, averaging, st_bin, 2.5*100, l)
println("The simulation for c = 0.01, T=$T, N=100, time steps = 100, l=$(l) is finished.")

a = averaged_density2(100, 0.01, T, 200, averaging, st_bin, 3*100, l)
println("The simulation for c = 0.01, T=$T, N=100, time steps = 200, l=$(l) is finished.")

a = averaged_density2(500, 0.002, T, 10, averaging, st_bin, 4*10, l)
println("The simulation for c = 0.002, T=$T, N=500, time steps = 10, l=$(l) is finished.")

a = averaged_density2(500, 0.002, T, 100, averaging, st_bin, 2.5*100, l)
println("The simulation for c = 0.002, T=$T, N=500, time steps = 100, l=$(l) is finished.")

a = averaged_density2(500, 0.002, T, 200, averaging, st_bin, 3*100, l)
println("The simulation for c = 0.002, T=$T, N=500, time steps = 200, l=$(l) is finished.")