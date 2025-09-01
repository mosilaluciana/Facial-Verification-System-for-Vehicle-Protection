/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */


#define TRIG_FRONT_PIN GPIO_PIN_9
#define TRIG_FRONT_PORT GPIOA
#define ECHO_FRONT_PIN GPIO_PIN_8
#define ECHO_FRONT_PORT GPIOA

#define TRIG_BACK_PIN GPIO_PIN_5
#define TRIG_BACK_PORT GPIOA
#define ECHO_BACK_PIN GPIO_PIN_6
#define ECHO_BACK_PORT GPIOA
#define OBSTACLE_LIMIT_CM 10

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */
uint8_t rxData;
char lastCommand = 0 ; // Ultima comanda primita

const uint32_t COMMAND_TIMEOUT = 500; // 500 ms timeout



uint32_t pMillis;
uint32_t Value1_Front = 0, Value2_Front = 0;
uint32_t Value1_Back = 0, Value2_Back = 0;
uint16_t Distance_Front = 0;
uint16_t Distance_Back = 0;

uint8_t obstacleDetected = 0;




/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM3_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */



void updateSensors() {
    // Trigger Front Sensor (TIM3)
    HAL_GPIO_WritePin(TRIG_FRONT_PORT, TRIG_FRONT_PIN, GPIO_PIN_SET);
    __HAL_TIM_SET_COUNTER(&htim3, 0);
    while (__HAL_TIM_GET_COUNTER(&htim3) < 10);
    HAL_GPIO_WritePin(TRIG_FRONT_PORT, TRIG_FRONT_PIN, GPIO_PIN_RESET);

    pMillis = HAL_GetTick();
    while (!(HAL_GPIO_ReadPin(ECHO_FRONT_PORT, ECHO_FRONT_PIN)) && pMillis + 10 > HAL_GetTick());
    Value1_Front = __HAL_TIM_GET_COUNTER(&htim3);

    pMillis = HAL_GetTick();
    while ((HAL_GPIO_ReadPin(ECHO_FRONT_PORT, ECHO_FRONT_PIN)) && pMillis + 50 > HAL_GetTick());
    Value2_Front = __HAL_TIM_GET_COUNTER(&htim3);

    Distance_Front = (Value2_Front - Value1_Front) * 0.034 / 2;

    // Trigger Back Sensor (TIM1)
    HAL_GPIO_WritePin(TRIG_BACK_PORT, TRIG_BACK_PIN, GPIO_PIN_SET);
    __HAL_TIM_SET_COUNTER(&htim1, 0);
    while (__HAL_TIM_GET_COUNTER(&htim1) < 10);
    HAL_GPIO_WritePin(TRIG_BACK_PORT, TRIG_BACK_PIN, GPIO_PIN_RESET);

    pMillis = HAL_GetTick();
    while (!(HAL_GPIO_ReadPin(ECHO_BACK_PORT, ECHO_BACK_PIN)) && pMillis + 10 > HAL_GetTick());
    Value1_Back = __HAL_TIM_GET_COUNTER(&htim1);

    pMillis = HAL_GetTick();
    while ((HAL_GPIO_ReadPin(ECHO_BACK_PORT, ECHO_BACK_PIN)) && pMillis + 50 > HAL_GetTick());
    Value2_Back = __HAL_TIM_GET_COUNTER(&htim1);

    Distance_Back = (Value2_Back - Value1_Back) * 0.034 / 2;

    if (Distance_Front < OBSTACLE_LIMIT_CM || Distance_Back < OBSTACLE_LIMIT_CM) {
        obstacleDetected = 1;
    } else {
        obstacleDetected = 0;
    }
}








// Motor A input (PB4 PB5) direction
 
 void setDirectionMotor_A_FWD(){ // 1 0
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, 1); 
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_4, 0); 
	}
 
void setDirectionMotor_A_BWD(){ // 0 1
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, 0); 
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_4, 1);
	}

void setDirectionMotor_A_STOP(){ 	
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, 0);
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_4, 0); 
	}
	
	
	// Motor B input (PA4 PB0) direction
void setDirectionMotor_B_FWD(){ // 1 0 
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, 1); 
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, 0);
}

void setDirectionMotor_B_BWD(){ // 0 1
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, 0); 
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, 1); 
}
void setDirectionMotor_B_STOP(){
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, 0); 
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, 0); 
	}
	
	
	
void controlMotors()
{
	
	
		if (obstacleDetected) {
					setDirectionMotor_A_STOP();
					setDirectionMotor_B_STOP();
					__HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, 0);
					__HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, 0);
					return;
			}

	
    if (lastCommand == 'w') // FWD
    {
        setDirectionMotor_A_FWD();
        setDirectionMotor_B_FWD();
    }
    else if (lastCommand == 's') // BWD
    {
        setDirectionMotor_A_BWD();
        setDirectionMotor_B_BWD();
    }
    else if (lastCommand == 'a') // LEFT
    {
        setDirectionMotor_A_FWD();
        setDirectionMotor_B_BWD();
    }
    else if (lastCommand == 'd') // RIGHT
    {
        setDirectionMotor_A_BWD();
        setDirectionMotor_B_FWD();
    }
    else // Stop motors when no key is pressed
    {
        setDirectionMotor_A_STOP();
        setDirectionMotor_B_STOP();
    }

    // Aplica viteza
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, (lastCommand ? 1000 : 0));
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, (lastCommand ? 1000 : 0));
}






/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART1_UART_Init();
  MX_TIM2_Init();
  MX_TIM1_Init();
  MX_TIM3_Init();
  /* USER CODE BEGIN 2 */
	 
	HAL_UART_Receive_IT(&huart1,&rxData,1); // Enabling interrupt receive
	
	HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3); //PB10 TIM2 CH3
	HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2); //PA1 TIM2 CH2
	
	
	setDirectionMotor_A_STOP();
	setDirectionMotor_B_STOP();
	
	
	HAL_TIM_Base_Start(&htim1);
	HAL_TIM_Base_Start(&htim3);
	HAL_GPIO_WritePin(TRIG_FRONT_PORT, TRIG_FRONT_PIN, GPIO_PIN_RESET);
	HAL_GPIO_WritePin(TRIG_BACK_PORT, TRIG_BACK_PIN, GPIO_PIN_RESET);
	
	HAL_UART_Receive_IT(&huart1, &rxData, 1);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
		updateSensors();	
		
		if (obstacleDetected) {
        lastCommand = 0;  // ignora comenzile daca e obstacol
    }
		controlMotors(); // Actualizeaza mi?carea ma?inu?ei
		HAL_Delay(100);
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  RCC_OscInitStruct.PLL.PREDIV = RCC_PREDIV_DIV1;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART1|RCC_PERIPHCLK_TIM1
                              |RCC_PERIPHCLK_TIM2|RCC_PERIPHCLK_TIM34;
  PeriphClkInit.Usart1ClockSelection = RCC_USART1CLKSOURCE_PCLK2;
  PeriphClkInit.Tim1ClockSelection = RCC_TIM1CLK_HCLK;
  PeriphClkInit.Tim2ClockSelection = RCC_TIM2CLK_HCLK;
  PeriphClkInit.Tim34ClockSelection = RCC_TIM34CLK_HCLK;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 71;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 1999;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 71;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 1999;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.Pulse = 999;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit(&htim2);

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 72;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 1999;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 9600;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4|GPIO_PIN_5|GPIO_PIN_9, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0|GPIO_PIN_4|GPIO_PIN_5, GPIO_PIN_RESET);

  /*Configure GPIO pins : PA4 PA5 PA9 */
  GPIO_InitStruct.Pin = GPIO_PIN_4|GPIO_PIN_5|GPIO_PIN_9;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PA6 PA8 */
  GPIO_InitStruct.Pin = GPIO_PIN_6|GPIO_PIN_8;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PB0 PB4 PB5 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_4|GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/*
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if(huart->Instance==USART1)
  {
    if(rxData=='w') // FWD 
    {
			setDirectionMotor_A_FWD();
			setDirectionMotor_B_FWD();
				
			__HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, 1000);
			__HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, 1000);
  
    }
    else if (rxData== 's') // BWD
    {
    	
			setDirectionMotor_A_BWD();
			setDirectionMotor_B_BWD();
				
			__HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, 1000);
			__HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, 1000);
    }
		else if (rxData== 'a') // LEFT
    {
    	setDirectionMotor_A_FWD();
				setDirectionMotor_B_BWD();
				
			__HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, 1000);
			__HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, 1000);
			
    }
		else if (rxData== 'd') // RIGHT
    {
    	setDirectionMotor_A_BWD(); 
			setDirectionMotor_B_FWD();
				
			__HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, 1000);
			__HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, 1000);
    }
    HAL_UART_Receive_IT(&huart1,&rxData,1); // Enabling interrupt receive again
  }
}
*/


void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1)
    {
        HAL_UART_Transmit(&huart1, &rxData, 1, 100); // feedback

        if (rxData == 'w' || rxData == 's' || rxData == 'a' || rxData == 'd') 
        {
            lastCommand = rxData; // Save movement command
        } 
        else if (rxData == ' ') 
        {
            lastCommand = 0; // Stop command
        }

        HAL_UART_Receive_IT(&huart1, &rxData, 1); // Re-enable UART reception
    }
}




/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
