/*
======================================================================
services/api.ts
======================================================================
Sends base64 image to backend, receives prediction, and returns
nutrition data by indexing a local JSON file.
======================================================================
*/

import nutritionData from '../assets/nutrition_data.json';

export interface NutritionInfo {
  Calories: string;
  Protein: string;
  Carbohydrates: string;
  Fat: string;
}

export interface FoodPrediction {
  name: string;
  confidence: number;
  nutrition: NutritionInfo;
}

const defaultNutrition: NutritionInfo = {
  Calories: 'N/A',
  Protein: 'N/A',
  Carbohydrates: 'N/A',
  Fat: 'N/A',
};

/**
 * Looks up nutrition data from JSON using the label.
 */
const getNutritionInfo = (label: string): NutritionInfo => {
  return nutritionData[label] ?? defaultNutrition;
};

/**
 * Sends base64 image to the backend for prediction.
 */
export const predictFood = async (base64Image: string): Promise<FoodPrediction> => {
  const apiUrl = process.env.EXPO_PUBLIC_API_URL;
  if (!apiUrl) throw new Error('Missing EXPO_PUBLIC_API_URL in .env');

  try {
    const response = await fetch(`${apiUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Image }),
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);

    const data = await response.json();
    console.log(data);
    const prediction: FoodPrediction = {
      name: data.label,
      confidence: data.confidence ?? 1.0,
      nutrition: getNutritionInfo(data.label),
    };

    return prediction;
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
};
