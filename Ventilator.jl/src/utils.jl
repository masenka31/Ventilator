function RandomBatch(xdata,ydata;batchsize::Int=64)
    l = length(xdata)
	if batchsize > l
		return (xdata,ydata)
	end
    idx = sample(1:l,batchsize)
    return xdata[idx],ydata[idx]
end
function RandomBatch(xdata,ydata,bdata;batchsize::Int=64)
    l = length(xdata)
	if batchsize > l
		return (xdata,ydata,bdata)
	end
    idx = sample(1:l,batchsize)
    return xdata[idx],ydata[idx],bdata[idx]
end