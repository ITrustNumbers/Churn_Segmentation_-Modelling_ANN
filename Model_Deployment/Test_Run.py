import json
import requests

def multiple(feature):
    out_data = json.dumps({'data':feature})
    response = requests.post(url+'/batch_predict',out_data)
    return response.json()

def single(feature):
    out_data = json.dumps({'data':feature})
    response = requests.post(url+'/predict',out_data)
    return response.json()

url = 'http://127.0.0.1:5000/'

number = int(input('For how many customers you need the prediction for: '))

print('\nInput data for {} customer'.format(number))

data = []

for n in range(number):

    print('\nFor Customer Number {}'.format(n+1))
    crscore = float(input('\nEnter Credit Score: '))
    geography = input('Geographic Location of customer can only be one Spain, Germany or France (Case Sensitive): ')
    gender = input('Gender of customer Male or Female (Case Sensitive): ')
    age = float(input('Age of Customer: '))
    tenure = float(input('Number of Tenure ongoing with the bank: '))
    balance = float(input('Remaining balance: '))
    Num = float(input('Number of Products of bank that the Customers uses(Out of 3): '))
    hascr = float(input('Whether the customer has Credit Card or not, If yes than use 1 else give 0: '))
    active = float(input('If the Customer is an active member of the bank give 1 else 0: '))
    salary = float(input('The Estimated Salary of the customer(only figure): '))

    data.append([crscore, geography, gender, age, tenure, balance, Num, hascr, active, salary])

if number != 1:
    result = multiple(data)
    preds = result['preds']
    probs = result['probs']
    
    for n in range(number):
        print('\nThe Prediction for customer number {}'.format(n))
        print(preds[n])
        print('The Respective Pribability: {:.4f}'.format(probs[n]))

    flag = intput('Press Enter to Exit')
else:
    result = single(data[0]) 
    print('\nThe Prediction for the given customer')
    print(result['pred'][0])
    print('The Respective Pribability: {:.4f}'.format(result['prob'][0]))
    flag = intput('Press Enter to Exit')