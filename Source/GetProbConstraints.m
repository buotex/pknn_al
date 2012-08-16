function C = GetProbConstraints(y, l, u)
%Computes the constraints for pknn classifier
m = length(y);
numclass=length(unique(y));
C = zeros(m*numclass, 4);
count=1;
for i=1:m
    for j=1:numclass
        if(j==y(i))
            C(count,:)=[i j 1 u];
            count=count+1;
        else
            C(count,:)=[i j -1 l];
            count=count+1;
        end
    end
end
C=C(1:count-1,:);