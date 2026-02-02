import torch
from HAM import HAMLoss

if __name__ == "__main__":

    H = 720
    V = 321

    x = torch.arange(H*V).reshape((1, H, V)).to(torch.device("cuda"))
    y = torch.ones((1, H, V)).to(torch.device("cuda"))
    
    import time
    
    for err in ["mape", "smape", "mase", "mse", "mae"]:

        if err in ["mse", "mae"]:
            epochs = 1000
        else:
            epochs = 100
    
        MSELoss1 = HAMLoss(H, loss_fn = err, device=torch.device("cuda"))
        MSELoss2 = HAMLoss(H, loss_fn = err, device=torch.device("cuda"), num_variates=V, variate_indices=list(range(1,V,V//75)))
        #MSELoss1 = HAMLoss(H, device=torch.device("cuda"))
        #MSELoss2 = HAMLoss(H, device=torch.device("cuda"), num_variates=V, variate_indices=list(range(1,V,V//75)))
        bef = time.time()
        for _ in range(epochs):
            for mode in ["causal", "anti-causal"]:
                for h in range(0, H+1):
                    MSELoss1.assign_mask(h, mode)
                    loss = MSELoss1(x, y)
                    #from pprint import pprint
                    #pprint (loss)
               
                    MSELoss2.assign_mask(h, mode)
                    loss = MSELoss2(x, y)
                    #from pprint import pprint
                    #pprint (loss)
            
        print ("%s GPU: %.5f sec per data pt" % (err, (time.time() - bef) / epochs))

        MSELoss1 = HAMLoss(H, device=torch.device("cuda"), cpu_gpu_transfer=True)
        MSELoss2 = HAMLoss(H, device=torch.device("cuda"), num_variates=V, variate_indices=(1,3), cpu_gpu_transfer=True)
        
        bef = time.time()
        for _ in range(epochs):
            for mode in ["causal", "anti-causal"]:
                for h in range(0, H+1):
     
                    MSELoss1.assign_mask(h, mode)
                    loss = MSELoss1(x, y)
                    #from pprint import pprint
                    #pprint (loss)

                    MSELoss2.assign_mask(h, mode)
                    loss = MSELoss2(x, y)
                    #from pprint import pprint
                    #pprint (loss)

        print ("%s CPU-GPU: %.5f sec per data pt\n" % (err, (time.time() - bef) / epochs))
