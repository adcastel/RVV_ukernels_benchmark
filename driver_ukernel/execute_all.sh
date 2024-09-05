#/bin/bash

ini=8
end=48
step=8
ininr=4
endnr=48
stepnr=4
mkdir -p out
for mode in LDX OPT BASE;
do
    for swap in 0 1;
      do
      if [ ${swap} -eq 1 ]; then
            mm="loadBA"
          else
            mm="loadAB"
          fi
        for gather in 0 1 2;
        do
	  if [ ${gather} -eq 1 ]; then
            gg="gather"
          else 
	    if [ ${gather} -eq 0 ]; then
              gg="bcast"
	    else
	      gg="macc"
	    fi
          fi
	  for r in 1024 2048;
          do
	    ff=out/${mode}_${mm}_${gg}_${r}.dat
	    echo "" > ${ff}
            for mr in $(seq ${ini} ${step} ${end});
            do
               for nr in $(seq ${ininr} ${stepnr} ${endnr});
               do 
		       make clean
		       make MR=${mr} NR=${nr} MODE=${mode} GATHER=${gather} SWAP=${swap}
		       ./test_uk ${mr} ${mr} ${nr} ${nr} ${r} 0 1000 >> ${ff}
	      done
	    done
         done
      done
   done
done

