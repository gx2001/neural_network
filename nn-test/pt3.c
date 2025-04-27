#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define D 2         //x1, x2
#define K 4         //C1, C2, C3, C4
#define H1 20      //Neurones gia epipedo 1
#define H2 25       //Neurones gia epipedo 2
#define H3 20      // Neurones gia epipedo 3
#define EPOCHS 5000
#define MIN_EPOCHS 800
#define THRESHOLD 1e-5
#define LR 0.01
#define BATCH_SIZE 20
#define USE_TANH 1  // 1 -> tanh() , 0 -> ReLU
#define TRAIN_SIZE 4000
#define TEST_SIZE  4000


int classify_point(float x1, float x2) {
    float distA = (x1 - 0.5f)*(x1 - 0.5f) + (x2 - 0.5f)*(x2 - 0.5f);
    float distB = (x1 + 0.5f)*(x1 + 0.5f) + (x2 + 0.5f)*(x2 + 0.5f);
    float distC = (x1 - 0.5f)*(x1 - 0.5f) + (x2 + 0.5f)*(x2 + 0.5f);
    float distD = (x1 + 0.5f)*(x1 + 0.5f) + (x2 - 0.5f)*(x2 - 0.5f);

    if(distA < 0.2f && x2 > 0.5f) return 0;
    if(distA < 0.2f && x2 < 0.5f) return 1;
    if(distB < 0.2f && x2 > -0.5f) return 0;
    if(distB < 0.2f && x2 < -0.5f) return 1;
    if(distC < 0.2f && x2 > -0.5f) return 0;
    if(distC < 0.2f && x2 < -0.5f) return 1;
    if(distD < 0.2f && x2 > 0.5f) return 0;
    if(distD < 0.2f && x2 < 0.5f) return 1;

    if(x1 * x2 > 0.0f) return 2;
    return 3;
}

//paragoume ta train_data.txt, test_data.txt
void generate_data() {
    FILE *ftrain = fopen("train_data.txt", "w");
    FILE *ftest  = fopen("test_data.txt",  "w");
    if(!ftrain || !ftest) {
        printf("Error opening output files.\n");
        exit(1);
    }

    fprintf(ftrain, "%d\n", TRAIN_SIZE);
    fprintf(ftest,  "%d\n", TEST_SIZE);

    int total = TRAIN_SIZE + TEST_SIZE;
    for(int i = 0; i < total; i++){
        float x1 = -1.0f + 2.0f*((float)rand()/RAND_MAX);
        float x2 = -1.0f + 2.0f*((float)rand()/RAND_MAX);
        int c = classify_point(x1, x2);

        if(i < TRAIN_SIZE) {
            fprintf(ftrain, "%.6f  %.6f  %d\n", x1, x2, c);
        } else {
            fprintf(ftest,  "%.6f  %.6f  %d\n", x1, x2, c);
        }
    }

    fclose(ftrain);
    fclose(ftest);
    printf("Data generation done! Created train_data.txt & test_data.txt.\n");
}

//fortosi dedomenwn 
void loadData(
    float **Xtrain, float **Ttrain, int *N,
    float **Xtest,  float **Ttest,  int *NVAL
){
    FILE *ftrain = fopen("train_data.txt", "r");
    FILE *ftest  = fopen("test_data.txt",  "r");
    if(!ftrain || !ftest) {
        printf("Error opening train/test files.\n");
        exit(1);
    }

    fscanf(ftrain, "%d", N);
    *Xtrain = (float *)malloc((*N)*D*sizeof(float));
    *Ttrain = (float *)malloc((*N)*K*sizeof(float));

    for(int i=0; i<*N; i++){
        float x1, x2;
        int c;
        fscanf(ftrain, "%f %f %d", &x1, &x2, &c);

        (*Xtrain)[i*D + 0] = x1;
        (*Xtrain)[i*D + 1] = x2;

        for(int kk=0; kk<K; kk++){
            (*Ttrain)[i*K + kk] = 0.0f;
        }
        if(c>=0 && c<K){
            (*Ttrain)[i*K + c] = 1.0f;
        }
    }

    fscanf(ftest, "%d", NVAL);
    *Xtest = (float *)malloc((*NVAL)*D*sizeof(float));
    *Ttest = (float *)malloc((*NVAL)*K*sizeof(float));

    for(int i=0; i<*NVAL; i++){
        float x1, x2;
        int c;
        fscanf(ftest, "%f %f %d", &x1, &x2, &c);

        (*Xtest)[i*D + 0] = x1;
        (*Xtest)[i*D + 1] = x2;

        for(int kk=0; kk<K; kk++){
            (*Ttest)[i*K + kk] = 0.0f;
        }
        if(c>=0 && c<K){
            (*Ttest)[i*K + c] = 1.0f;
        }
    }

    fclose(ftrain);
    fclose(ftest);
}
//domes me 3 krifa epipeda 

float W1[H1][D],  b1[H1];
float W2[H2][H1], b2[H2];
float W3[H3][H2], b3[H3];
float W4[K][H3],  b4[K];   // output epipedo 

// forward pass
float z1[H1], a1[H1];
float z2[H2], a2[H2];
float z3[H3], a3[H3];
float z4[K],  a4[K];  // teliki eksodos 

//ReLU / tanh
static inline float relu(float x) {
    return (x>0.0f) ? x : 0.0f;
}
static inline float d_relu(float x) {
    return (x>0.0f) ? 1.0f : 0.0f;
}
static inline float d_tanhf(float val) {
    // val = tanh(u) -> val' = 1 - val^2
    return (1.0f - val*val);
}

//Forward pass me 3 epipeda

void forward_pass(const float *x, int d, float *y, int k) {
    // 1 epipedo
    for(int i=0; i<H1; i++){
        float sum_ = 0.0f;
        for(int j=0; j<d; j++){
            sum_ += W1[i][j] * x[j];
        }
        sum_ += b1[i];
        z1[i] = sum_;
   
        a1[i] = tanhf(z1[i]);
    }

    // 2 epipedo
    for(int i=0; i<H2; i++){
        float sum_ = 0.0f;
        for(int j=0; j<H1; j++){
            sum_ += W2[i][j] * a1[j];
        }
        sum_ += b2[i];
        z2[i] = sum_;
        a2[i] = tanhf(z2[i]);

    }

    // 3 epipedo
    for(int i=0; i<H3; i++){
        float sum_ = 0.0f;
        for(int j=0; j<H2; j++){
            sum_ += W3[i][j] * a2[j];
        }
        sum_ += b3[i];
        z3[i] = sum_;
    #if USE_TANH
        a3[i] = tanhf(z3[i]);
    #else
        a3[i] = relu(z3[i]);
    #endif
    }

    // Final epipedo
    for(int i=0; i<k; i++){
        float sum_ = 0.0f;
        for(int j=0; j<H3; j++){
            sum_ += W4[i][j] * a3[j];
        }
        sum_ += b4[i];
        z4[i] = sum_;

        a4[i] = tanhf(z4[i]);   // grammatiki eksodos 

    }

    // y[]
    for(int i=0; i<k; i++){
        y[i] = a4[i];
    }
}

// Backprop me 3 epipeda 
static float dW1[H1][D],  db1[H1];
static float dW2[H2][H1], db2[H2];
static float dW3[H3][H2], db3[H3];
static float dW4[K][H3],  db4[K];

void backprop(const float *x, int d, const float *t, int k) {
    // Error eksodou
    float delta4[K];
    for(int i=0; i<k; i++){
        float diff = a4[i] - t[i];

        delta4[i] = diff * d_tanhf(a4[i]);

    }

    // 3o epipedo
    float delta3[H3];
    for(int i=0; i<H3; i++){
        float sum_ = 0.0f;
        for(int j=0; j<k; j++){
            sum_ += W4[j][i] * delta4[j];
        }
    #if USE_TANH
        delta3[i] = sum_ * d_tanhf(a3[i]);
    #else
        delta3[i] = sum_ * d_relu(z3[i]);
    #endif
    }

    // 2o epipedo
    float delta2[H2];
    for(int i=0; i<H2; i++){
        float sum_ = 0.0f;
        for(int j=0; j<H3; j++){
            sum_ += W3[j][i] * delta3[j];
        }

        delta2[i] = sum_ * d_tanhf(a2[i]);
    }

    //1o epipedo
    float delta1[H1];
    for(int i=0; i<H1; i++){
        float sum_ = 0.0f;
        for(int j=0; j<H2; j++){
            sum_ += W2[j][i] * delta2[j];
        }
        delta1[i] = sum_ * d_tanhf(a1[i]);

    }

    //update gia ta W4, b4
    for(int i=0; i<k; i++){
        for(int j=0; j<H3; j++){
            dW4[i][j] += delta4[i] * a3[j];
        }
        db4[i] += delta4[i];
    }

    // Update W3, b3
    for(int i=0; i<H3; i++){
        for(int j=0; j<H2; j++){
            dW3[i][j] += delta3[i] * a2[j];
        }
        db3[i] += delta3[i];
    }

    // Update W2, b2
    for(int i=0; i<H2; i++){
        for(int j=0; j<H1; j++){
            dW2[i][j] += delta2[i] * a1[j];
        }
        db2[i] += delta2[i];
    }

    // Update W1, b1
    for(int i=0; i<H1; i++){
        for(int j=0; j<d; j++){
            dW1[i][j] += delta1[i] * x[j];
        }
        db1[i] += delta1[i];
    }
}

//MAIN 
int main() {
    srand((unsigned)time(NULL));

    // Dimiourgia dedomenon
    generate_data();

    // Fortosi dedomenon
    float *Xtrain=NULL, *Ttrain=NULL;
    float *Xtest=NULL,  *Ttest=NULL;
    int N=0, NVAL=0;
    loadData(&Xtrain, &Ttrain, &N, &Xtest, &Ttest, &NVAL);
    printf("Loaded train samples: %d, test samples: %d\n", N, NVAL);

    //Tixea arxikopihsi
    for(int i=0; i<H1; i++){
        for(int j=0; j<D; j++){
            W1[i][j] = 2.0f*((float)rand()/RAND_MAX)-1.0f;
        }
        b1[i] = 2.0f*((float)rand()/RAND_MAX)-1.0f;
    }
    for(int i=0; i<H2; i++){
        for(int j=0; j<H1; j++){
            W2[i][j] = 2.0f*((float)rand()/RAND_MAX)-1.0f;
        }
        b2[i] = 2.0f*((float)rand()/RAND_MAX)-1.0f;
    }
    for(int i=0; i<H3; i++){
        for(int j=0; j<H2; j++){
            W3[i][j] = 2.0f*((float)rand()/RAND_MAX)-1.0f;
        }
        b3[i] = 2.0f*((float)rand()/RAND_MAX)-1.0f;
    }
    for(int i=0; i<K; i++){
        for(int j=0; j<H3; j++){
            W4[i][j] = 2.0f*((float)rand()/RAND_MAX)-1.0f;
        }
        b4[i] = 2.0f*((float)rand()/RAND_MAX)-1.0f;
    }

    // loss file
    FILE *floss = fopen("loss_per_epoch.txt", "w");
    if(!floss){
        printf("Error opening loss file.\n");
        exit(1);
    }

    //
    float prev_error = 1e9;
    for(int epoch=0; epoch<EPOCHS; epoch++){
        // Midenismos gradients
        for(int i=0; i<H1; i++){
            for(int j=0; j<D; j++){
                dW1[i][j] = 0.0f;
            }
            db1[i] = 0.0f;
        }
        for(int i=0; i<H2; i++){
            for(int j=0; j<H1; j++){
                dW2[i][j] = 0.0f;
            }
            db2[i] = 0.0f;
        }
        for(int i=0; i<H3; i++){
            for(int j=0; j<H2; j++){
                dW3[i][j] = 0.0f;
            }
            db3[i] = 0.0f;
        }
        for(int i=0; i<K; i++){
            for(int j=0; j<H3; j++){
                dW4[i][j] = 0.0f;
            }
            db4[i] = 0.0f;
        }

        float total_loss = 0.0f;

        //loop
        for(int nIdx=0; nIdx<N; nIdx++){
            float *x = &Xtrain[nIdx*D];
            float *t = &Ttrain[nIdx*K];
            float y[K];
            forward_pass(x, D, y, K);

            // MSE
            float e=0.0f;
            for(int i=0; i<K; i++){
                float diff = (y[i]-t[i]);
                e += diff*diff;
            }
            total_loss += 0.5f* e; //loss

            // Backprop
            backprop(x, D, t, K);

            // Mini-batch update
            if(((nIdx+1)%BATCH_SIZE==0)||(nIdx==N-1)){
                int batch_count = ((nIdx+1)%BATCH_SIZE==0)? BATCH_SIZE: (nIdx% BATCH_SIZE+1);

                // Update W1,b1
                for(int i=0; i<H1; i++){
                    for(int j=0; j<D; j++){
                        W1[i][j] -= (LR/batch_count)*dW1[i][j];
                        dW1[i][j]=0.0f;
                    }
                    b1[i] -= (LR/batch_count)*db1[i];
                    db1[i]=0.0f;
                }
                // Update W2,b2
                for(int i=0; i<H2; i++){
                    for(int j=0; j<H1; j++){
                        W2[i][j] -= (LR/batch_count)*dW2[i][j];
                        dW2[i][j]=0.0f;
                    }
                    b2[i] -= (LR/batch_count)*db2[i];
                    db2[i]=0.0f;
                }
                // Update W3,b3
                for(int i=0; i<H3; i++){
                    for(int j=0; j<H2; j++){
                        W3[i][j] -= (LR/batch_count)*dW3[i][j];
                        dW3[i][j]=0.0f;
                    }
                    b3[i] -= (LR/batch_count)*db3[i];
                    db3[i]=0.0f;
                }
                // Update W4,b4
                for(int i=0; i<K; i++){
                    for(int j=0; j<H3; j++){
                        W4[i][j] -= (LR/batch_count)*dW4[i][j];
                        dW4[i][j]=0.0f;
                    }
                    b4[i] -= (LR/batch_count)*db4[i];
                    db4[i]=0.0f;
                }
            }
        }

        float avg_loss = total_loss / N;
        printf("Epoch %d | MSE = %f\n", epoch, avg_loss);
        fprintf(floss, "%d  %.6f\n", epoch, avg_loss);

        // Stop criterion
        if(epoch>MIN_EPOCHS){
            float diff = fabs(prev_error - avg_loss);
            if(diff<THRESHOLD){
                printf("Stop training: MSE change < %g\n", THRESHOLD);
                break;
            }
        }
        prev_error=avg_loss;
    }
    fclose(floss);

    //Test accuracy
    int correct_count=0;
    for(int nIdx=0; nIdx<NVAL; nIdx++){
        float *x = &Xtest[nIdx*D];
        float *t = &Ttest[nIdx*K];
        float y[K];
        forward_pass(x, D, y, K);

        int pc=0;
        float mv=y[0];
        for(int i=1; i<K; i++){
            if(y[i]>mv){
                mv=y[i];
                pc=i;
            }
        }
        int tc=0;
        for(int i=0; i<K; i++){
            if(t[i]>0.5f){
                tc=i;
                break;
            }
        }
        if(pc==tc){
            correct_count++;
        }
    }
    float accuracy = 100.0f* correct_count/ (float)NVAL;
    printf("\nTest Accuracy = %.2f%%\n", accuracy);

    //Taksinomisi tou arxiou
    FILE *fout = fopen("classified_test_points.txt","w");
    if(fout){
        for(int nIdx=0; nIdx<NVAL; nIdx++){
            float *x = &Xtest[nIdx*D];
            float *t = &Ttest[nIdx*K];
            float y[K];
            forward_pass(x, D, y, K);

            int pc=0;
            float mv=y[0];
            for(int i=1; i<K; i++){
                if(y[i]>mv){
                    mv=y[i];
                    pc=i;
                }
            }
            int tc=0;
            for(int i=0; i<K; i++){
                if(t[i]>0.5f){
                    tc=i;
                    break;
                }
            }
            int correct=(pc==tc);
            fprintf(fout,"%.5f  %.5f  %d  %d  %d\n", x[0], x[1], tc, pc, correct);
        }
        fclose(fout);
        printf("Wrote classified_test_points.txt\n");
    }

    //ekatharisi mnimis 
    free(Xtrain);
    free(Ttrain);
    free(Xtest);
    free(Ttest);

    return 0;
}
