function f(x)
    x + 2
end

function g(x)
    3 * x
end

function h(x)
    f(g(x))
end

map(h, [2, 3, 4, 5])