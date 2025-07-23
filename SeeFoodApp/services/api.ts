/*
======================================================================
services/api.ts
======================================================================
This file contains the fake API call. It now includes fake
nutritional information for each food item.
*/

// Define an interface for the nutritional data
export interface NutritionInfo {
    Calories: string;
    Protein: string;
    Carbohydrates: string;
    Fat: string;
}

// Update the main prediction interface to include nutrition
export interface FoodPrediction {
    name: string;
    confidence: number;
    nutrition: NutritionInfo;
}

// A list of possible predictions for our fake API, now with nutrition.
const fakeFoodPredictions: FoodPrediction[] = [
    { 
        name: 'Pizza', 
        confidence: 0.98, 
        nutrition: { Calories: '285 kcal', Protein: '12g', Carbohydrates: '36g', Fat: '10g' } 
    },
    { 
        name: 'Sushi', 
        confidence: 0.92, 
        nutrition: { Calories: '45 kcal', Protein: '3g', Carbohydrates: '7g', Fat: '0.5g' } 
    },
    { 
        name: 'Burger', 
        confidence: 0.95, 
        nutrition: { Calories: '540 kcal', Protein: '30g', Carbohydrates: '45g', Fat: '29g' } 
    },
    { 
        name: 'Pasta', 
        confidence: 0.88, 
        nutrition: { Calories: '220 kcal', Protein: '8g', Carbohydrates: '43g', Fat: '1.5g' } 
    },
    { 
        name: 'Salad', 
        confidence: 0.99, 
        nutrition: { Calories: '150 kcal', Protein: '5g', Carbohydrates: '10g', Fat: '10g' } 
    },
    { 
        name: 'Tacos', 
        confidence: 0.94, 
        nutrition: { Calories: '226 kcal', Protein: '12g', Carbohydrates: '18g', Fat: '11g' } 
    },
];

/**
 * Simulates predicting a food item from a base64 image string.
 * @param {string} base64Image - The base64 encoded image string.
 * @returns {Promise<FoodPrediction>} A promise that resolves with the prediction.
 */
export const predictFood = (base64Image: string): Promise<FoodPrediction> => {
  console.log('Sending image data to the server for analysis...');

  return new Promise(resolve => {
    // Simulate a network delay of 2 seconds
    setTimeout(() => {
      // Return a random food item from our list
      const randomPrediction = fakeFoodPredictions[Math.floor(Math.random() * fakeFoodPredictions.length)];
      console.log('Prediction received:', randomPrediction);
      resolve(randomPrediction);
    }, 2000);
  });
};
