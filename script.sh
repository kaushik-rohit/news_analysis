threshold=(0.1 0.2 0.4 0.5 0.6 0.7 0.8 0.9)

for i in "${threshold[@]}";do
    python analysis.py -d ./Nexis -v ./vec.pkl -a source -t $i
    python analysis.py -d ./Nexis -v ./vec.pkl -t $i
done
