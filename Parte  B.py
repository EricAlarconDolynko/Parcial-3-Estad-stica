import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.stats import t


def generar_muestras(delta: int):
    """
    Genera las muestras i.i.d de X, Y con media Delta y segundo parámetro 1
    Ambas de tamaño 60. 
    """
    return np.random.laplace(loc=delta, scale=1, size=60)

### BOOTSTRAP ###

def generar_bootstrap(x:list, y:list, iteraciones:int):
    
    z = []
    
    for dato in x:
        z.append(dato)
    for dato in y:
        z.append(dato)
    
    lista_bootsrap = []
    for i in range(iteraciones):
        nuevo_x = []
        for i in range(60):
            indice = random.randint(0,len(z)-1)
            nuevo_x.append(z[indice])
        lista_bootsrap.append(nuevo_x)
    
    return lista_bootsrap

def prueba_laplace_delta(delta:int, bootstrap:int):
    
    x = generar_muestras(0)
    y = generar_muestras(delta)
    
    t_observada = np.median(y) - np.median(x)
    x_bootstrap = generar_bootstrap(x, y, bootstrap)
    y_bootstrap = generar_bootstrap(x, y, bootstrap)
    
    counter = 0
    lista_t_bootstrap = []
    for i in range(bootstrap):
        t_bootstrap = np.median(y_bootstrap[i]) - np.median(x_bootstrap[i])
        if t_bootstrap >= t_observada:
            counter += 1
        lista_t_bootstrap.append(t_bootstrap)
        
    t_sorted = sorted(lista_t_bootstrap)
    c_alfa = t_sorted[int((bootstrap*0.95) - 1)]
    
    flag = False
    
    if c_alfa >= t_observada:
        print(f"ACEPTA LA HIPOTESIS NULA CON {c_alfa} y {t_observada}")
        flag = True
    else:
        flag = False
        print(f"RECHAZA LA HIPOTESIS NULA CON {c_alfa} y {t_observada}")
        
    p_value_gorro = counter/bootstrap 
    
    return flag

def generar_muestras_t_student(k: int, delta:int):
    """
    Generar las muestras i.i.d de tamaño 60 de una t-student con k grados de libertado.
    """
    return delta + np.random.standard_t(df=k, size=60)
    

def prueba_t_student_delta(delta:int, grados:int, bootstrap:int):
    x = generar_muestras_t_student(grados, 0)
    y = generar_muestras_t_student(grados, delta)
    
    t_observada = np.median(y) - np.median(x)
    x_bootstrap = generar_bootstrap(x, y, bootstrap)
    y_bootstrap = generar_bootstrap(x, y, bootstrap)
    
    counter = 0
    lista_t_bootstrap = []
    for i in range(bootstrap):
        t_bootstrap = np.median(y_bootstrap[i]) - np.median(x_bootstrap[i])
        if t_bootstrap >= t_observada:
            counter += 1
        lista_t_bootstrap.append(t_bootstrap)
    
    t_sorted = sorted(lista_t_bootstrap)
    c_alfa = t_sorted[int((bootstrap*0.95) - 1)]
    
    flag = False
    
    if c_alfa >= t_observada:
        print(f"ACEPTA LA HIPOTESIS NULA CON {c_alfa} y {t_observada}")
        flag = True
    else:
        flag = False
        print(f"RECHAZA LA HIPOTESIS NULA CON {c_alfa} y {t_observada}")
            
    p_value_gorro = counter/bootstrap 
    
    return flag

### FIN BOOTSTRAP ###

### WILCOXON ###

def combinar_y_determinar_rangos(x:list, y:list):
    combinacion = []
    for dato in x:
        combinacion.append(dato)
    for dato in y:
        combinacion.append(dato)
    
    organizado = sorted(combinacion)
    return organizado

def calcular_wilcoxon(combinado: list, y: list):
    suma = 0
    for dato in y:
        suma += combinado.index(dato)
    return suma

def prueba_wilcoxon_laplace(delta: int):
    x = generar_muestras(0)
    y = generar_muestras(delta)
    
    combinacion = combinar_y_determinar_rangos(x,y)
    wilcoxon = calcular_wilcoxon(combinacion, y)
    
    flag = False
    
    if wilcoxon >= 3943:
        flag = False
        print(f"Se RECHAZA la Hipotesis pues W tiene valor de: {wilcoxon}")
    else:
        print(f"Se ACEPTA la Hipotesis pues W tiene valor de : {wilcoxon}")
        flag = True
    
    return flag
    
    
def prueba_wilcoxon_t_student(delta: int, grados: int):
    x = generar_muestras_t_student(grados, 0)
    y = generar_muestras_t_student(grados, delta)
    
    combinacion = combinar_y_determinar_rangos(x,y)
    wilcoxon = calcular_wilcoxon(combinacion, y)
    
    flag = False
    
    if wilcoxon >= 3943:
        flag = False
        print(f"Se RECHAZA la Hipotesis pues W tiene valor de: {wilcoxon}")
    else:
        print(f"Se ACEPTA la Hipotesis pues W tiene valor de : {wilcoxon}")
        flag = True
    
    return flag
    

### FIN WILCOXON ###

### T - COMBINADO ###

def promedio_muestral(muestra:list):
    suma = 0
    for dato in muestra:
        suma += dato
    return suma / (len(muestra))
    

def calcular_S(muestra: list):
    n = len(muestra)
    promedio = promedio_muestral(muestra)
    suma = 0
    for i in range(n):
        suma += (muestra[i]-promedio)**2
    s = ((1)/(n-1))*suma
    return s
        
def calcular_Sp(x:list, y:list):
    s1 = calcular_S(x)
    s2 = calcular_S(y)
    n = len(x)
    m = len(y)
    numerador = ((n-1)*s1) + ((m-1)*s2)
    denominador = n+m-2
    return numerador/denominador

def calcular_t_combinado(x: list, y: list):
    x_barra = promedio_muestral(x)
    y_barra = promedio_muestral(y)
    n = len(x)
    m = len(y)
    numerador = x_barra - y_barra
    denominador = math.sqrt(calcular_Sp(x,y)) * math.sqrt( (1/n) + (1/m) )
    return numerador/denominador

def prueba_combinado_laplace(delta: int):
    x = generar_muestras(0)
    y = generar_muestras(delta)
    t_combinado = abs(calcular_t_combinado(x,y))
    
    flag = False
    
    if t_combinado >= 1.657:
        flag = False
        print(f"RECHAZAR Hipótesis con t_combinado = {t_combinado}")
    else:
        print(f"ACEPTAR Hipótesis con t_combinado = {t_combinado}")
        flag = True
        
    return flag

def prueba_combinado_t_student(delta:int, grados: int):
    x = generar_muestras_t_student(grados, 0)
    y = generar_muestras_t_student(grados, delta)
    t_combinado = abs(calcular_t_combinado(x,y))
    
    flag = False
    
    if t_combinado >= 1.657:
        flag = False
        print(f"RECHAZAR Hipótesis con t_combinado = {t_combinado}")
    else:
        print(f"ACEPTAR Hipótesis con t_combinado = {t_combinado}")
        flag = True
        
    return flag
    
        
### FIN T - COMBINADO ###

### CONCLUSION ###

def contar_errores_delta_0(iteraciones: int):
    categorias = ["Laplace", "t2", "t10", "t50"]
    mediana = [0, 0, 0, 0]
    wilcoxon = [0, 0, 0, 0]
    t_combinado = [0, 0, 0, 0]

    for i in range(iteraciones):
        
        mediana_1 = prueba_laplace_delta(0, 10000)
        if not mediana_1:
            mediana[0] += 1
        mediana_2 = prueba_t_student_delta(0, 2, 10000)
        if not mediana_2:
            mediana[1] += 1
        mediana_3 = prueba_t_student_delta(0, 10, 10000)
        if not mediana_3:
            mediana[2] += 1
        mediana_4 = prueba_t_student_delta(0, 50, 10000)
        if not mediana_4:
            mediana[3] += 1
        
        wilcoxon_1 = prueba_wilcoxon_laplace(0)
        if not wilcoxon_1:
            wilcoxon[0] += 1
        wilcoxon_2 = prueba_wilcoxon_t_student(0,2)
        if not wilcoxon_2:
            wilcoxon[1] += 1
        wilcoxon_3 = prueba_wilcoxon_t_student(0,10)
        if not wilcoxon_3:
            wilcoxon[2] += 1
        wilcoxon_4 = prueba_wilcoxon_t_student(0, 50)
        if not wilcoxon_4:
            wilcoxon[3] += 1
        
        t_combinado1 = prueba_combinado_laplace(0)
        if not t_combinado1:
            t_combinado[0] += 1
        t_combinado2 = prueba_combinado_t_student(0, 2)
        if not t_combinado2:
            t_combinado[1] += 1
        t_combinado3 = prueba_combinado_t_student(0, 10)
        if not t_combinado3:
            t_combinado[2] += 1
        t_combinado4 = prueba_combinado_t_student(0, 50)
        if not t_combinado4:
            t_combinado[3] += 1
                
        print(f"Iteración delta 0 {i+1}")
        
    
    datos = np.array([mediana, wilcoxon, t_combinado]) 

    n_categorias = len(categorias)
    n_estadisticos = datos.shape[0]

    x = np.arange(n_categorias)

    ancho = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(n_estadisticos):
        ax.bar(x + i*ancho, datos[i], width=ancho, label=f'Estadístico {i+1}')

    ax.set_xlabel('Pruebas')
    ax.set_ylabel('Cantidad de errores  cuando Delta = 0')
    ax.set_title('Errores por estadístico en cada prueba')
    ax.set_xticks(x + ancho)  
    ax.set_xticklabels(categorias)
    ax.legend()
    plt.show()


def contar_errores_delta_3(iteraciones: int):
    categorias = ["Laplace", "t2", "t10", "t50"]
    mediana = [0, 0, 0, 0]
    wilcoxon = [0, 0, 0, 0]
    t_combinado = [0, 0, 0, 0]

    for i in range(iteraciones):
        
        mediana_1 = prueba_laplace_delta(0.3, 10000)
        if mediana_1:
            mediana[0] += 1
        mediana_2 = prueba_t_student_delta(0.3, 2, 10000)
        if mediana_2:
            mediana[1] += 1
        mediana_3 = prueba_t_student_delta(0.3, 10, 10000)
        if mediana_3:
            mediana[2] += 1
        mediana_4 = prueba_t_student_delta(0.3, 50, 10000)
        if mediana_4:
            mediana[3] += 1
        
        wilcoxon_1 = prueba_wilcoxon_laplace(0.3)
        if wilcoxon_1:
            wilcoxon[0] += 1
        wilcoxon_2 = prueba_wilcoxon_t_student(0.3,2)
        if wilcoxon_2:
            wilcoxon[1] += 1
        wilcoxon_3 = prueba_wilcoxon_t_student(0.3,10)
        if wilcoxon_3:
            wilcoxon[2] += 1
        wilcoxon_4 = prueba_wilcoxon_t_student(0.3, 50)
        if wilcoxon_4:
            wilcoxon[3] += 1
        
        t_combinado1 = prueba_combinado_laplace(0.3)
        if t_combinado1:
            t_combinado[0] += 1
        t_combinado2 = prueba_combinado_t_student(0.3, 2)
        if t_combinado2:
            t_combinado[1] += 1
        t_combinado3 = prueba_combinado_t_student(0.3, 10)
        if t_combinado3:
            t_combinado[2] += 1
        t_combinado4 = prueba_combinado_t_student(0.3, 50)
        if t_combinado4:
            t_combinado[3] += 1
                
        print(f"Iteración delta 3 {i+1}")
        
    
    datos = np.array([mediana, wilcoxon, t_combinado]) 

    n_categorias = len(categorias)
    n_estadisticos = datos.shape[0]

    x = np.arange(n_categorias)

    ancho = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(n_estadisticos):
        ax.bar(x + i*ancho, datos[i], width=ancho, label=f'Estadístico {i+1}')

    ax.set_xlabel('Pruebas')
    ax.set_ylabel('Cantidad de errores cuando Delta = 0.3')
    ax.set_title('Errores por estadístico en cada prueba')
    ax.set_xticks(x + ancho)  
    ax.set_xticklabels(categorias)
    ax.legend()
    plt.show()

### FIN CONCLUSION ###


### EJECUCION ###

#laplace = prueba_laplace_delta(0.3, 10000)
#print(f"p_gorro = {laplace}")
#t_student = prueba_t_student_delta(0.3,50, 10000)
#print(f"p_gorro = {t_student}")

#prueba_wilcoxon_laplace(0)
#prueba_wilcoxon_t_student(0, 50)

#prueba_combinado_laplace(0)
#prueba_combinado_t_student(0, 50)

#contar_errores_delta_0(1000)
#contar_errores_delta_3(1000)

#print(t.ppf(0.95, 118))