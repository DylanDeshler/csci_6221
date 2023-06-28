# Basic concurrent problem with 4 workers and variable number of tasks, prints how long each slept for

const jobs = Channel{Int}(32)
const results = Channel{Tuple}(32)

function work()
    for id in jobs
        exec_time = rand()
        sleep(exec_time)

        put!(results, (id, exec_time))
    end
end

function make_jobs(n)
    for i in 1:n
        put!(jobs, i)
    end
end

n = 8
errormonitor(@async make_jobs(n))

for i in 1:4
    errormonitor(@async work())
end

@elapsed while n > 0
    id, exec_time = take!(results)
    println("$id finished in $(round(exec_time; digits=2)) seconds")
    global n = n -1
end