import joblib
import pandas as pd




# Load the trained Random Forest model
model_RF = joblib.load("stress_model_RF.joblib")
model_LR = joblib.load("stress_model_LR.joblib")
model_KNN = joblib.load("stress_model_KNN.joblib")

print("Dine værdier skal nu indtastes")
sleep_Duration = float(input("Indtast søvnlængde i timer: "))
quality_of_sleep = int(input("Indtast kvaliteten af din søvn mellem 1 og 10, hvor 1 er dårligt og 10 er godt: "))
gender_numerical = int(input("Indtast biologisk køn (0 for kvinde, 1 for mand): "))


physical_activity_level_before_scaling = int(input("Indtast fysisk aktivitet mellem 1 og 100: "))
# Scaler 'Physical Activity Level' colonnen, så den er mellem 1-10
physical_activity_level_scaled = 1 + (physical_activity_level_before_scaling - 1) * (10 - 1) / (100 - 1)


heart_rate_before_scaling = int(input("Indtast hvilepuls (antal slag pr. minut): "))
# Scaler 'Heart Rate' colonnen, så den er mellem 1-10
heart_rate_scaled = 1 + (heart_rate_before_scaling - 1) * (10 - 1) / (100 - 1)

daily_steps_before_scaling = int(input("Indtast antal daglige skridt: "))
# Scaler 'Daily Steps' colonnen, så den er mellem 1-10
daily_steps_scaled = 1 + (daily_steps_before_scaling - 1) * (10 - 1) / (10000 - 1)

age_before_scaling = int(input("Indtast alder: "))
# Scaler 'Age' colonnen, så den er mellem 1-10
age_scaled = 1 + (age_before_scaling - 1) * (10 - 1) / (100 - 1)

#Ekstra steps for BMI kategorierne
# Hjemmeside brugt som BMI skema https://www.klinik.dk/dit-sundhedstjek/bmi-klassificering-skema/
weight = float(input("Indtast vægt: "))
height = float(input("Indtast højde: "))
bmi_raw_value = weight / ((height/100)**2)

bmi = []
if bmi_raw_value < 18.5:
    bmi = 0  #Underweight
elif 18.5 <= bmi_raw_value <= 24.9:
    bmi = 1  #Normal weight
elif 25 <= bmi_raw_value <= 29.9:
    bmi = 2  #Overweight
elif bmi_raw_value >= 30:
    bmi = 3  #Obese



#Definér nye inputdata (erstat med dine faktiske inputværdier)
new_data = pd.DataFrame({
    "Sleep Duration": [sleep_Duration],
    "Quality of Sleep": [quality_of_sleep],
    "Weight_Category_BMI": [bmi],
    "Gender_Numerical": [gender_numerical],
    "Physical Activity Level (scaled)": [physical_activity_level_scaled],
    "Heart Rate (scaled)": [heart_rate_scaled],
    "Daily Steps (scaled)": [daily_steps_scaled],
    "Age (scaled)": [age_scaled],
})

#Laver prediction af de nye data
predictions_RF = model_RF.predict(new_data)
predictions_LR = model_LR.predict(new_data)
predictions_KNN = model_KNN.predict(new_data)
predictions = predictions_RF + predictions_LR + predictions_KNN



#Printer simpelt output af de nye data. 
#Predictions er ML-modellens output givet som 
if predictions >= 2:
    print("Du er stresset, læs yderligere for at finde ud af, hvad du kan gøre, for at nedbringe dine stress faktorer")
    if sleep_Duration < 6:
        print("Dit søvnniveau er lavt, det er anbefaldet af sundhedsstyrelsen at voksne mennesker helst skal have 7-9 timer søvn pr. dag. 6 og 10 timer kan også være en passende søvnlængde for nogle mennesker")
    #https://www.sst.dk/-/media/Udgivelser/2024/Soevn/Anbefalinger-for-soevnlaengde.ashx
    if quality_of_sleep < 6:
        print("Din søvnkvalitet er lav, bedre søvnkvalitet, kan være at sove længere eller kortere, man skal efter sin søvn føle sig udhvilet. Prøv at få en bedre døgnrytme eller udmatte kroppen før sengetid")
    #https://netdoktor.dk/soevn/sygdomme/sovn-hvad-er-normalt/
    if heart_rate_scaled > 8:
        print("En normal hvilepuls for mænd er 60-80, kvinder 70-90 en høj hvilepuls er derfor ikke nødvendigvis farlig, men træning kan få ens hvilepuls ned. Ved hvilepuls forhøjning over tid, kan dette være tegn på stress")
    #https://hjerteforeningen.dk/artikler/alle-artikler/dette-skal-du-vide-om-pulsen-og-dit-hjerte/
    if daily_steps_scaled < 8000:
        print("Sigt gerne efter at ramme mindst 7000 skridt, hjerteforeningen viser at du har 50-70 procent lavere dødlighed vedrørende udvikling af hjerte-kar-sygdomme ved at gøre dette. Dog findes der intet magisk tal, det handler mest om, hvor aktiv du er i din dagligdag.")
    #https://hjerteforeningen.dk/artikler/alle-artikler/saa-mange-skridt-skal-du-gaa-for-et-laengere-liv/
    if physical_activity_level_scaled < 6:
        print("Din fysiske aktivitet er lav, sigt efter at være aktiv 30 minutter om dagen med mindst 2 fysiske træninger med høj intensitet. Dette er dog mindste værdierne, så at træne yderligere giver sundhedsmæssige fordele")
    #https://www.sundhed.dk/borger/forebyggelse/livsstil/fysisk-aktivitet/
    if bmi == 0:
        print("Du er undervægtig hvilket kan øge risikoen for at udvikle stress og andre sundhedsproblemer. Dog tager en BMI måling ikke højdre for muskel og fedtmasse, BMI bruges mest som en general målefaktor på den generalle befolkning")
    if bmi == 2:
        print("Du er overvægtig, hvilket kan øge risikoen for at udvikle stress og andre sundhedsproblemer. Dog tager en BMI måling ikke højdre for muskel og fedtmasse, BMI bruges mest som en general målefaktor på den generalle befolkning")
    if bmi == 3:
        print("Du er svært overvægtig, hvilket kan øge risikoen for at udvikle stress og andre sundhedsproblemer. Dog tager en BMI måling ikke højdre for muskel og fedtmasse, BMI bruges mest som en general målefaktor på den generalle befolkning")
    #https://www.nature.com/articles/s42003-023-05396-8
else:
    print("Du er ikke stresset")


input("Press Enter to exit...")
