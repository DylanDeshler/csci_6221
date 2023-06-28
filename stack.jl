import Base: push!, pop!, length, isempty

mutable struct Stack{T}
    data::Vector{T}
end

function Stack{T}() where T
    Stack{T}(T[])
end

function push!(stack::Stack{T}, item::T) where T
    push!(stack.data, item)
end

function pop!(stack::Stack{T}) where T
    if isempty(stack.data)
        error("Stack is empty")
    else
        pop!(stack.data)
    end
end

function top(stack::Stack{T}) where T
    if isempty(stack.data)
        error("Stack is empty")
    else
        stack.data[end]
    end
end

function isempty(stack::Stack{T}) where T
    isempty(stack.data)
end

function length(stack::Stack{T}) where T
    length(stack.data)
end

stack = Stack{Int}()
push!(stack, 1)
push!(stack, 2)
push!(stack, 3)

println("Length ", length(stack))
println("Top: ", top(stack))

for i in 1:length(stack)
    println(pop!(stack))
end

println(isempty(stack))