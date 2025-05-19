import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstwo

### PUNTO NEWTON - RAPHSON ###

def generar_muestras(n:int):
    """ 
    Función que genera muestras para n = 50, 100, 200, 500 y 1000
    
    Donde Se generan 1000 muestras i.i.d de tamaño n
    """
    return np.random.logistic(loc=0, scale=1, size=(1000,n))

def promedio_muestral(muestra: list):
    suma = 0
    for dato in muestra:
        suma += dato
    return suma/len(muestra)
        
def S_de_teta_k(muestra: int, theta_k: int):
    n = len(muestra)
    suma = 0
    for i in range(n):
        numerador = math.exp(-(muestra[i]-theta_k))
        denominador = 1 + math.exp(-(muestra[i]-theta_k))
        suma += (1 - (2*(numerador/denominador)))
        
    return suma
        
def H_de_teta_k(muestra: int, theta_k: int):
    n = len(muestra)
    suma = 0
    for i in range(n):
        numerador = 2*math.exp(-(muestra[i]-theta_k))
        denominador = (1 + math.exp(-(muestra[i]-theta_k)))**2
        suma += (numerador/denominador)
    
    return suma
        
def newton_raphson(muestra: list, iteraciones: int):
    """
    Función para encontrar el emv Teta, tomando como
    punto incial el promedio de los datos.
    """
    lista_thetas = []
    theta_k = promedio_muestral(muestra)
    lista_thetas.append(theta_k)
    for i in range(iteraciones):
        theta_next = theta_k + (S_de_teta_k(muestra, theta_k)/H_de_teta_k(muestra,theta_k))
        lista_thetas.append(theta_next)
        theta_k = theta_next
    
    return lista_thetas
        
def graficar_emv(muestras: list):
    """ 
    Grafíca el resultado del emv calculado por Newton-Raphson para las 1000 muestras
    con un n fijo.
    """ 
    matriz_thetas = []
    media = 0
    i = 1
    
    for muestra in muestras:
        matriz_thetas.append(newton_raphson(muestra, 1000))
        print(f"Iteración: {i}")
        i += 1
    
    matriz_thetas = np.array(matriz_thetas)  
    estimadores_finales = matriz_thetas[:, -1] 
    
    emv_theta = promedio_muestral(estimadores_finales)
    print(f"EMV THETA = {emv_theta}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(estimadores_finales, bins=30, density=True, alpha=0.7, edgecolor='black')
    plt.title('Histograma de emv en múltiples muestras')
    plt.xlabel('Valor EMV')
    plt.ylabel('Densidad')
    plt.grid(True)
    plt.show()

### FIN PUNTO NEWTON - RAPHSON ###


### PUNTO ESTADISTICOS CON EL EMV ###

def log_verosimilitud(muestra: list, theta: float):
    suma = 0
    n = len(muestra)
    for i in range(n):
        suma += ( (-(muestra[i]-theta)) - 2*math.log(1 + math.exp(-(muestra[i]-theta)) ) )
        
    return suma

def graficar_likelihood_ratio(muestras:list):
    lista_likelihoods = []
    i = 1
    for muestra in muestras:
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        likelihood = -2*(log_verosimilitud(muestra, 0) - log_verosimilitud(muestra, emv))
        lista_likelihoods.append(likelihood)
        print(f"Iteración: {i}")
        i += 1
    
    plt.figure(figsize=(10, 6))
    plt.hist(lista_likelihoods, bins=30, density=True, alpha=0.7, edgecolor='black')
    
    x = np.linspace(0, max(lista_likelihoods), 500)
    y = chi2.pdf(x, df=1)
    plt.plot(x, y, 'r--', label=r'Densidad $\chi^2_1$')
    
    plt.title('Histograma de Likelihood Ratio')
    plt.xlabel('Valor Estadistico Likelihood')
    plt.ylabel('Densidad')
    plt.grid(True)
    plt.show()
    
    
def graficar_Wald(muestras:list):
    lista_wald = []
    i = 1
    for muestra in muestras:
        n = len(muestra)
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        wald = (n* (emv**2))/3
        lista_wald.append(wald)
        print(f"Iteración: {i}")
        i+=1
        
    plt.figure(figsize=(10, 6))
    plt.hist(lista_wald, bins=30, density=True, alpha=0.7, edgecolor='black')
    
    x = np.linspace(0, max(lista_wald), 500)
    y = chi2.pdf(x, df=1)
    plt.plot(x, y, 'r--', label=r'Densidad $\chi^2_1$')
    
    plt.title('Histograma de Wald')
    plt.xlabel('Valor Estadistico Wald')
    plt.ylabel('Densidad')
    plt.grid(True)
    plt.show()
    
    
def graficar_rao_score(muestras:list):
    lista_rao = []
    i = 1
    for muestra in muestras:
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        rao = (S_de_teta_k(muestra, 0)**2)/(1*H_de_teta_k(muestra, emv))
        lista_rao.append(rao)
        print(f"Iteración: {i}")
        i+=1
        
    plt.figure(figsize=(10, 6))
    plt.hist(lista_rao, bins=30, density=True, alpha=0.7, edgecolor='black')
    
    x = np.linspace(0, max(lista_rao), 500)
    y = chi2.pdf(x, df=1)
    plt.plot(x, y, 'r--', label=r'Densidad $\chi^2_1$')
    
    plt.title('Histograma de Rao')
    plt.xlabel('Valor Estadistico Rao Score')
    plt.ylabel('Densidad')
    plt.grid(True)
    plt.show()
    
### FIN PUNTO ESTADISTICOS CON EL EMV ###
    

### PUNTO PEARSON DE ESTADISTICOS ###

def obtener_celdas_equiprobables():
    cuantiles = [chi2.ppf(q, df=1) for q in np.linspace(0, 1, 8)]
    
    for i in range(7):
        primer = chi2.cdf(cuantiles[i+1], df=1) - chi2.cdf(cuantiles[i], df=1)
        #print(f"Probabilidad soporte {i+1} = {primer}")
    
    return cuantiles

def crear_diccionario(celdas: list):
    diccionario = {}
    for i in range(7):
        diccionario[(celdas[i], celdas[i+1])] = 0
    return diccionario
        
def actualizar_diccionario(diccionario: dict, likelihood: float):
    
    for llave in diccionario:
        if llave[0] <= likelihood <= llave[1]:
            diccionario[llave] = diccionario[llave] + 1

def calcular_pearson_likelihood(muestras:list):
    celdas = obtener_celdas_equiprobables()
    diccionario = crear_diccionario(celdas)
    for muestra in muestras:
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        likelihood = -2*(log_verosimilitud(muestra, 0) - log_verosimilitud(muestra, emv))
        actualizar_diccionario(diccionario, likelihood)
        
    freq_esperada = len(muestras)/7
    pearson = 0
    for llave in diccionario:
        pearson += ((diccionario[llave]-freq_esperada)**2)/freq_esperada
        
    return pearson        
        
def graficar_pearson_likelihood():
    lista_pearson = []
    for i in range(1000):
        muestras = generar_muestras(1000)
        pearson = calcular_pearson_likelihood(muestras)
        lista_pearson.append(pearson)
        print(f"Iteración: {i+1}")
        i+=1
        
    plt.figure(figsize=(10, 6))
    plt.hist(lista_pearson, bins=30, density=True, alpha=0.7, edgecolor='black')
    
    x = np.linspace(0, max(lista_pearson), 500)
    y = chi2.pdf(x, df=6)
    plt.plot(x, y, 'r--', label=r'Densidad $\chi^2_6$')
    
    plt.title('Histograma de Pearson')
    plt.xlabel('Valor Pearson para Likelihood')
    plt.ylabel('Densidad')
    plt.grid(True)
    plt.show()
    
    
def calcular_pearson_wald(muestras:list):
    celdas = obtener_celdas_equiprobables()
    diccionario = crear_diccionario(celdas)
    for muestra in muestras:
        n = len(muestra)
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        wald = (n* (emv**2))/3
        actualizar_diccionario(diccionario, wald)
        
    freq_esperada = len(muestras)/7
    pearson = 0
    for llave in diccionario:
        pearson += ((diccionario[llave]-freq_esperada)**2)/freq_esperada
        
    return pearson

    
def graficar_pearson_wald():
    lista_pearson = []
    for i in range(1000):
        muestras = generar_muestras(1000)
        pearson = calcular_pearson_wald(muestras)
        lista_pearson.append(pearson)
        print(f"Iteración: {i+1}")
        i+=1
        
    plt.figure(figsize=(10, 6))
    plt.hist(lista_pearson, bins=30, density=True, alpha=0.7, edgecolor='black')
    
    x = np.linspace(0, max(lista_pearson), 500)
    y = chi2.pdf(x, df=6)
    plt.plot(x, y, 'r--', label=r'Densidad $\chi^2_6$')
    
    plt.title('Histograma de Pearson')
    plt.xlabel('Valor Pearson para Wald')
    plt.ylabel('Densidad')
    plt.grid(True)
    plt.show()
    
def calcular_pearson_rao(muestras:list):
    celdas = obtener_celdas_equiprobables()
    diccionario = crear_diccionario(celdas)
    for muestra in muestras:
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        rao = (S_de_teta_k(muestra, 0)**2)/(1*H_de_teta_k(muestra, emv))
        actualizar_diccionario(diccionario, rao)
        
    freq_esperada = len(muestras)/7
    pearson = 0
    for llave in diccionario:
        pearson += ((diccionario[llave]-freq_esperada)**2)/freq_esperada
        
    return pearson

    
def graficar_pearson_rao():
    lista_pearson = []
    for i in range(1000):
        muestras = generar_muestras(1000)
        pearson = calcular_pearson_rao(muestras)
        lista_pearson.append(pearson)
        print(f"Iteración: {i+1}")
        i+=1
        
    plt.figure(figsize=(10, 6))
    plt.hist(lista_pearson, bins=30, density=True, alpha=0.7, edgecolor='black')
    
    x = np.linspace(0, max(lista_pearson), 500)
    y = chi2.pdf(x, df=6)
    plt.plot(x, y, 'r--', label=r'Densidad $\chi^2_6$')
    
    plt.title('Histograma de Pearson')
    plt.xlabel('Valor Pearson para Rao Score')
    plt.ylabel('Densidad')
    plt.grid(True)
    plt.show()
    
### FIN PUNTO PEARSON DE ESTADISTICOS ###

### KOLMOGOROV-SMIRNOV ###

def mayor_diferencia(ordenados: list):
    max_dif = 0
    n = len(ordenados)
    for i in range(n):
        local_mas = abs( (i/n) - chi2.cdf(ordenados[i], df=1)  )
        local_menos = abs( chi2.cdf(ordenados[i], df=1) - ((i-1)/n)  )
        
        if local_mas > max_dif:
            max_dif = local_mas
        elif local_menos > max_dif:
            max_dif = local_menos
    
    return max_dif
        
    
def calcular_kolmogorov_likelihood(muestras: list):
    lista_kolmogorov = []
    i = 1
    for muestra in muestras:
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        likelihood = -2*(log_verosimilitud(muestra, 0) - log_verosimilitud(muestra, emv))
        lista_kolmogorov.append(likelihood)
        #print(f"Iteración: {i}")
        i+=1
        
    ordenada = sorted(lista_kolmogorov)
    max_dif = mayor_diferencia(ordenada)
    
    return max_dif    

def calcular_kolmogorov_wald(muestras:list):
    lista_kolmogorov = []
    i=1
    for muestra in muestras:
        n = len(muestra)
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        wald = (n* (emv**2))/3
        lista_kolmogorov.append(wald)
        #print(f"Iteración: {i}")
        i+=1
        
    ordenada = sorted(lista_kolmogorov)
    max_dif = mayor_diferencia(ordenada)
    
    return max_dif    

def calcular_kolmogorov_rao(muestras:list):
    lista_kolmogorov = []
    i = 1
    for muestra in muestras:
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        rao = (S_de_teta_k(muestra, 0)**2)/(1*H_de_teta_k(muestra, emv))
        lista_kolmogorov.append(rao)
        #print(f"Iteración: {i}")
        i+=1
        
    ordenada = sorted(lista_kolmogorov)
    max_dif = mayor_diferencia(ordenada)
    
    return max_dif    


### FIN KOLMOGOROV-SMIRNOV ###

### CRAMER-VON-MISES ###

def mayor_diferencia_cramer(ordenada: list):
    n = len(ordenada)
    suma = 0
    for i in range(1, n+1):
        suma += ( (chi2.cdf(ordenada[i-1], df=1) - ( ( (2*i) - 1) / (2*n) ) ) ** 2)
        
    estadistico = ( (1/n) * suma ) + (1 / (12*n))
    return estadistico
    

def calcular_cramer_likelihood(muestras: list):
    lista_cramer = []
    i = 1
    for muestra in muestras:
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        likelihood = -2*(log_verosimilitud(muestra, 0) - log_verosimilitud(muestra, emv))
        lista_cramer.append(likelihood)
        #print(f"Iteración: {i}")
        i+=1
        
    ordenada = sorted(lista_cramer)
    max_dif = mayor_diferencia_cramer(ordenada)
    #print(f"La maxima diferencia es {max_dif}")
    return max_dif   


def calcular_cramer_wald(muestras: list):
    lista_cramer = []
    i = 1
    for muestra in muestras:
        n = len(muestra)
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        wald = (n* (emv**2))/3
        lista_cramer.append(wald)
        #print(f"Iteración: {i}")
        i+=1
        
    ordenada = sorted(lista_cramer)
    max_dif = mayor_diferencia_cramer(ordenada)
    #print(f"La maxima diferencia es {max_dif}")
    return max_dif   

def calcular_cramer_rao(muestras: list):
    lista_cramer = []
    i = 1
    for muestra in muestras:
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        rao = (S_de_teta_k(muestra, 0)**2)/(1*H_de_teta_k(muestra, emv))
        lista_cramer.append(rao)
        #print(f"Iteración: {i}")
        i+=1
        
    ordenada = sorted(lista_cramer)
    max_dif = mayor_diferencia_cramer(ordenada)
    #print(f"La maxima diferencia es {max_dif}")
    return max_dif   

### FIN CRAMER-VON-MISES ###

### ANDERSON DARLING ###

def estadistico_anderson(ordenados: list):
    suma = 0
    n = len(ordenados)
    for i in range(1, n+1):
        suma += (2*i - 1) * ( math.log(chi2.cdf(ordenados[i-1], df=1))  +  math.log(1 - chi2.cdf(ordenados[n+1-i-1], df=1))  )
    
    estadistico = (-1* n) - ( (1/n) * suma )
    return estadistico        

def calcular_anderson_likelihood(muestras:list):
    lista_anderson = []
    i = 1
    for muestra in muestras:
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        likelihood = -2*(log_verosimilitud(muestra, 0) - log_verosimilitud(muestra, emv))
        lista_anderson.append(likelihood)
        #print(f"Iteración: {i}")
        i+=1
        
    ordenada = sorted(lista_anderson)
    anderson = estadistico_anderson(ordenada)
    #print(f"El estadistico es {anderson}")
    return anderson   

def calcular_anderson_wald(muestras:list):
    lista_anderson = []
    i = 1
    for muestra in muestras:
        n = len(muestra)
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        wald = (n* (emv**2))/3
        lista_anderson.append(wald)
        #print(f"Iteración: {i}")
        i+=1
        
    ordenada = sorted(lista_anderson)
    anderson = estadistico_anderson(ordenada)
    #print(f"El estadistico es {anderson}")
    return anderson   

def calcular_anderson_rao(muestras:list):
    lista_anderson = []
    i = 1
    for muestra in muestras:
        newton = newton_raphson(muestra, 1000)
        emv = newton[-1]
        rao = (S_de_teta_k(muestra, 0)**2)/(1*H_de_teta_k(muestra, emv))
        lista_anderson.append(rao)
        #print(f"Iteración: {i}")
        i+=1
        
    ordenada = sorted(lista_anderson)
    anderson = estadistico_anderson(ordenada)
    #print(f"El estadistico es {anderson}")
    return anderson   
    
### FIN ANDERSON DARLING ###

### EVALUAR PRUEBAS DE BONDAD ### 

def evaluar_bondad_suave(test: str, estadistico: str, c_alfa: float, n:int, iteraciones: int):
    valores = []
    if test == "kolmogorov":
        if estadistico == "likelihood":
            for i in range(iteraciones):
                muestras = generar_muestras(n)
                valores.append(calcular_kolmogorov_likelihood(muestras))
                print(f"Iteracion {i+1}")
                
        elif estadistico == "wald":
            for i in range(iteraciones):
                muestras = generar_muestras(n)
                valores.append(calcular_kolmogorov_wald(muestras))
                print(f"Iteracion {i+1}")
            
        elif estadistico == "rao":
            for i in range(iteraciones):
                muestras = generar_muestras(n)
                valores.append(calcular_kolmogorov_rao(muestras))
                print(f"Iteracion {i+1}")
                       
    elif test == "cramer":
        
        if estadistico == "likelihood":
            for i in range(iteraciones):
                muestras = generar_muestras(n)
                valores.append(calcular_cramer_likelihood(muestras))
                print(f"Iteracion {i+1}")
                
        elif estadistico == "wald":
            for i in range(iteraciones):
                muestras = generar_muestras(n)
                valores.append(calcular_cramer_wald(muestras))
                print(f"Iteracion {i+1}")
                
        elif estadistico == "rao":
            for i in range(iteraciones):
                muestras = generar_muestras(n)
                valores.append(calcular_cramer_rao(muestras))
                print(f"Iteracion {i+1}")
                
    elif test == "anderson":
        if estadistico == "likelihood":
            for i in range(iteraciones):
                muestras = generar_muestras(n)
                valores.append(calcular_anderson_likelihood(muestras))
                print(f"Iteracion {i+1}")
                
        elif estadistico == "wald":
            for i in range(iteraciones):
                muestras = generar_muestras(n)
                valores.append(calcular_anderson_wald(muestras))
                print(f"Iteracion {i+1}")
                
        elif estadistico == "rao":
            for i in range(iteraciones):
                muestras = generar_muestras(n)
                valores.append(calcular_anderson_rao(muestras))
                print(f"Iteracion {i+1}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(valores, bins=30, density=True, alpha=0.7, edgecolor='black')
    
    plt.axvline(x=c_alfa, color='red', linestyle='--', linewidth=2, label=f'cₐ = {c_alfa:.4f}')
    
    plt.title(f'Histograma con umbral crítico para la prueba {test} con  {estadistico}')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True)
    plt.show()

### FIN EVALUAR PRUEBAS DE BONDAD ### 

#EJECUCION#
n = 1000

#muestras = generar_muestras(n)
#graficar_emv(muestras)

#graficar_likelihood_ratio(muestras)
#graficar_Wald(muestras)        
#graficar_rao_score(muestras)

#graficar_pearson_likelihood()
#graficar_pearson_wald()
#graficar_pearson_rao()


#c_alpha = kstwo.ppf(0.95, n)
#calcular_kolmogorov_likelihood(muestras)   
#calcular_kolmogorov_wald(muestras)
#calcular_kolmogorov_rao(muestras)


#c_alpha = 0.461 
#calcular_cramer_likelihood(muestras)
#calcular_cramer_wald(muestras)
#calcular_cramer_rao(muestras)
  
#c_alpha = 2.492
#calcular_anderson_likelihood(muestras)
#calcular_anderson_wald(muestras)
#calcular_anderson_rao(muestras)
    
#evaluar_bondad_suave("anderson", "likelihood", c_alpha, n, 1000)
#evaluar_bondad_suave("anderson", "wald", c_alpha, n, 1000)
#evaluar_bondad_suave("anderson", "rao", c_alpha, n, 1000)



