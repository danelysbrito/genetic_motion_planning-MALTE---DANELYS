import random
import time
#from simulated_annealing_ok import load_and_create_environments as sa_load_and_create
from genetic_algorithm_MALTE_GA1 import load_and_create_environments as ga_load_and_create
from genetic_GA2_NO import load_and_create_environments as dane_load_and_create
from genetic_algorithm_DANELYS_GA3 import load_and_create_environments as ga_prueba_load_and_create

if __name__ == "__main__":
    random_seed = random.random() #devuelve un número de punto flotante pseudoaleatorio en el rango [0.0, 1.0)
    #random_seed_prueba = int(random.random() * 1000)
    print(f"SEMILLA ALEATORIA: {random_seed}")
    print("=======================================================")

    
    print("GA_1")
    start_time_m = time.time()
    ga_load_and_create("./environment1.yaml", random_seed)
    end_time_m = time.time()
    execution_time_m = end_time_m - start_time_m
    print("TIEMPO DE EJECUCION (ms) GA_1:", execution_time_m)

    print("=========================================================================")

    print("GA_2 :")
    start_time = time.time()
    dane_load_and_create("./environment1.yaml", random_seed)
    end_time = time.time()
    execution_time = end_time - start_time
    print("TIEMPO DE EJECUCION (ms) GA_2:", execution_time)

    print("=========================================================================")

    print("GA_3:")
    start_time_p = time.time()
    ga_prueba_load_and_create("./environment1.yaml", random_seed)
    end_time_p = time.time()
    execution_time_p = end_time_p - start_time_p
    print("TIEMPO DE EJECUCION (ms) GA_3:", execution_time_p)

    print("=========================================================================")

   
    
