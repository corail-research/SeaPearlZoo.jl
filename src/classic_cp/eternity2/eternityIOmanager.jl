struct EternityInputData
    n    ::Int
    m    ::Int
    pieces ::Matrix{Int}
end


function parseEternityInput(raw_input; order=[1,2,3,4])
    lines = split(raw_input, '\n')

    firstLine = split(lines[1], ' ')

    n = parse(Int, firstLine[1])
    m = parse(Int, firstLine[2])

    pieces = Matrix{Int}(undef, n*m, 4)

    for i = 2:n*m+1
        line = split(lines[i],' ')
        for j =1:4
            pieces[i-1,j] = parse(Int, line[order[j]])
        end
    end

    return EternityInputData(n, m, pieces)
end


function getEternityInputData(filename;order=[1,2,3,4])

    inputData = nothing

    open(filename, "r") do openedFile
        inputData = parseEternityInput(read(openedFile, String);order=order)
    end

    return inputData
end


"""
    print_eternity2(sol::Array{Int,3})

print a solution

# Arguments
- 'sol::Array{Int,3}' : one solution from OutputDataEternityII
"""
function print_eternity2(sol::Array{Int,3})
    id = sol[:,:,1]
    u = sol[:,:,2]
    r = sol[:,:,3]
    d = sol[:,:,4]
    l = sol[:,:,5]
    n = size(id,1)
    m = size(id,2)
    for k in 1:9*m
        print("-")
    end
    println()
    for i in 1:n
        print("|")
        for j in 1:m
            printstyled("   "*string(u[i,j],pad=2)*"   ", color=u[i,j])
            print("|")
        end
        println()
        print("|")
        for j in 1:m
            printstyled(string(l[i,j],pad=2),color=l[i,j])
            printstyled(" "*string(id[i,j],pad=2)*" ")
            printstyled(string(r[i,j],pad=2),color=r[i,j])
            print("|")
        end
        println()
        print("|")
        for j in 1:m
            printstyled("   "*string(d[i,j],pad=2)*"   ", color=d[i,j])
            print("|")
        end
        println()
        for k in 1:9*m
            print("-")
        end
        println()
    end
end


"""
    print_eternity2(model::SeaPearl.CPModel;limit=1)

print a certain number of solution

# Arguments
- 'model::SeaPearl.CPModel' : model solved
- 'limit::Int' : number of solutions to print
"""
function print_eternity2(model::SeaPearl.CPModel;limit=1)
    output = outputFromSeaPearl(model)
    print_eternity2(output;limit=limit)
end

function print_eternity2_fast_orientation(model::SeaPearl.CPModel;limit=1)
    output = outputFromSeaPearl_fast_orientation(model)
    print_eternity2(output;limit=limit)
end
