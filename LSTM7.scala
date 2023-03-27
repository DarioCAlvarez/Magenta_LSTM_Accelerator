import spatial.dsl._
import sys.process._

@spatial object LSTM7 extends SpatialApp {
   type T = FixPt[TRUE,_24,_8]    
   def approx_sig(x: T): T = {
     if(x > 2) {
       1.to[T]
     }
     else if(x <= -2) {
       0.to[T]
     }
     else {
       ((x / 4) + 0.5)
     }
   }

   def approx_tanh(x: T): T = {
     if(x > 1) {
       1.to[T]
     }
     else if(x <= -1) {
       -1.to[T]
     }
     else {
       x
     }
   }

   def main(args: Array[String]): Unit = {

     val weightsPath = "/home/dconnora/spatial_lab/apps/src/formatted_weights.csv"
     val xPath = "/home/dconnora/spatial_lab/apps/src/x.csv"
     val biasPath = "/home/dconnora/spatial_lab/apps/src/bias.csv" 

     val outerPar = 2
     val innerPar = 2
 
     val tileM = 16
     val tileN = 16
     val tileK = 16

     val M = ArgIn[Int]
     val N = ArgIn[Int]
     val K = ArgIn[Int]
     val Vsize = ArgIn[Int]

     //IMPORTANT NOTE: EITHER YOU NEED TO CALL THE FOLLOWING COMMAND MANUALLY EACH TIME YOU CHANGE THE WEIGHTS.CSV FILE OR YOU NEED TO RECOMPILE THE ENTIRE PROGRAM
     //THIS ONLY RUNS ONCE UPON RECOMPILE FOR SOME REASON
     val ran = "python3 /home/dconnora/spatial_lab/apps/src/format_weights.py" !!
     val a_data = loadCSV2D[T](weightsPath, ",", "\n")
     val x_data = loadCSV2D[T](xPath, ",", "\n")
     //Assume b_data is xt and h(t-1) vectors stacked together (4 elems each)
     val b_data = (0::a_data.cols, 0::x_data.cols){(i,j) => if (i < x_data.rows) x_data(i, j) else 0.to[T]}
     val c_init = (0::a_data.rows, 0::1.to[Int]){(i,j) => 0.to[T]}
    
     val vsize = x_data.rows
     val bsize = a_data.rows
     val tmax = x_data.cols


     setArg(M, a_data.rows)
     setArg(N, tmax.to[Int])
     setArg(K, a_data.cols)
     setArg(Vsize, vsize)


     //Bias simulation vectors
     val bi = loadCSV1D[T](biasPath, "\n")
     val ct_init = Array.tabulate(vsize) {i => 0.to[T] }

     //Matrices
     val a = DRAM[T](M, K)
     val b = DRAM[T](K, N)
     val c = DRAM[T](M, N)
     val ct = DRAM[T](Vsize)
     val out = DRAM[T](Vsize, tmax)
 
     //Biases
     val bias = DRAM[T](bsize)

     setMem(a, a_data)
     setMem(b, b_data)
     setMem(c, c_init)
     setMem(ct, ct_init)
     
     setMem(bias, bi)

     Accel {
       Sequential.Foreach(tmax by 1) {t =>
        Foreach(M by tileM /*par 4*/) { mm =>
         val numel_m = min(tileM.to[Int], M - mm)
         val tileC = SRAM[T](tileM, 1)
         Foreach(numel_m by 1/* par 4*/) { m =>
           tileC(m, 0) = 0.to[T]
         }
         c(mm::mm+numel_m, 0::1) store tileC
       }
      
       if(t != 0) {
       Foreach(Vsize by tileM /*par 4*/) {vv =>
         val numel_v = min(tileM.to[Int], Vsize - vv)
         val htPrev = SRAM[T](tileM, 1)
         htPrev load out(vv::vv+numel_v, t-1::t)
         b(vsize+vv::vsize+vv+numel_v, t::t+1) store htPrev
       }
      }

       /** Stage 1 **/
       //Matrix multiplication simultaneously adds xt and h(t-1)
       //val ran2 = "python3 /home/dconnora/spatial_lab/apps/src/time.py" !!
       Foreach(K by tileK par outerPar){kk =>
         val numel_k = min(tileK.to[Int], K - kk)
         Foreach(M by tileM par innerPar){mm =>
           val numel_m = min(tileM.to[Int], M - mm)
           val tileA_sram = SRAM[T](tileM, tileK)
           tileA_sram load a(mm::mm+numel_m, kk::kk+numel_k)
           //Do it as an array instead for N
           //Foreach(N by tileN par innerPar){nn =>
             //val numel_n = min(tileN.to[Int], N - nn)
             val numel_n = 1
             val tileB_sram = SRAM[T](tileK, numel_n)
             val tileC_sram = SRAM[T](tileM, numel_n).buffer
             tileB_sram load b(kk::kk+numel_k,t::t+numel_n)
             tileC_sram load c(mm::mm+numel_m, 0::numel_n)
 
  
             MemFold(tileC_sram /* par 4*/)(numel_k by 1/* par 4*/) { k =>
               val tile_temp = SRAM[T](tileM, numel_n)
               Foreach(numel_m by 1 par 4) {i =>
                 tile_temp(i, 0) = tileA_sram(i, k) * tileB_sram(k, 0)
               }
               tile_temp
             }{_+_}
             //printSRAM2(tileC_sram)
             c(mm::mm+numel_m, 0::numel_n) store tileC_sram
           //}
         }
       }
       //val ran3 = "python3 /home/dconnora/spatial_lab/apps/src/time.py" !!

     /** Stage 2 **/
     //Add bias weights to all mults
     //Call functions on the elements
     //Note we index the ram by 1 because it will always be by 1 (since N = 1)
     Foreach(M by tileM /*par 4*/) {mm =>
      val numel_m = min(tileM.to[Int], M - mm)
      val c_sram = SRAM[T](tileM, 1).buffer
      val bias_sram = SRAM[T](tileM).buffer
      val result = SRAM[T](tileM, 1).buffer
      c_sram load c(mm::mm+numel_m, 0::1)
      bias_sram load bias(mm::mm+numel_m)


      Foreach(numel_m by 1 /*par 4*/) {i =>
        if(i >= vsize && i < 2 * vsize) {
          result(i, 0) = approx_tanh(bias_sram(i) + c_sram(i, 0))
        }
        else {
          result(i, 0) = approx_sig(bias_sram(i) + c_sram(i, 0))
        }
      }
      c(mm::mm+numel_m, 0::1) store result

     }

     /** Stage 3-6 **/
     Foreach(Vsize by tileM /*par 4*/) {v =>
       val numel_v = min(tileM.to[Int], Vsize - v)
       val i_sram = SRAM[T](tileM, 1).buffer
       val c_sram = SRAM[T](tileM, 1).buffer
       val f_sram = SRAM[T](tileM, 1).buffer
       val o_sram = SRAM[T](tileM, 1).buffer
       val ctPrev = SRAM[T](tileM)
       val ctNew = SRAM[T](tileM)
       val htNew = SRAM[T](tileM, 1)

       i_sram load c(v::v+numel_v, 0::1)
       c_sram load c(v+vsize::v+vsize+numel_v, 0::1)
       f_sram load c(v+(2*vsize)::v+(2*vsize)+numel_v, 0::1)
       ctPrev load ct(v::v+numel_v)
       o_sram load c(v+(3*vsize)::v+(3*vsize)+numel_v, 0::1)

       Foreach(numel_v by  1  par 4) { i =>
         ctNew(i) = (i_sram(i, 0) * c_sram(i, 0)) + (f_sram(i, 0) * ctPrev(i))
         htNew(i, 0) = approx_tanh(ctNew(i)) * o_sram(i, 0)
       }
       ct(v::v+numel_v) store ctNew
       out(v::v+numel_v, t::t+1) store htNew
     }
       }
     }
     
 
     val accel_matrix = getMatrix(out)

     writeCSV2D(accel_matrix, "accel_mat.csv", ",", "\n") 
     printMatrix(accel_matrix, "Received: ")
   }
 }
















