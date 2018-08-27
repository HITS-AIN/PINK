#!groovy

pipeline {

  agent {
    label 'docker-gpu-host'
  }

  options {
    timeout(time: 1, unit: 'HOURS')
  }  

  stages {
    stage('Build') {
      parallel {
        stage('gcc-5') {
          agent {
            docker {
              reuseNode true
              image 'braintwister/ubuntu-16.04-cuda-9.2-cmake-3.12-gcc-5-conan-1.6'
              args '--runtime=nvidia'
            }
          }
          steps {
            sh './build.sh gcc-5'
          }
          post {
            always {
              step([
                $class: 'WarningsPublisher', canComputeNew: false, canResolveRelativePaths: false,
                defaultEncoding: '', excludePattern: '', healthy: '', includePattern: '', messagesPattern: '',
                parserConfigurations: [[parserName: 'GNU Make + GNU C Compiler (gcc)', pattern: 'build-gcc-5/make.out']],
                unHealthy: ''
              ])
            }
          }
        }
        stage('gcc-7') {
          agent {
            docker {
              reuseNode true
              image 'braintwister/ubuntu-16.04-cuda-9.2-cmake-3.12-gcc-7-conan-1.6'
              args '--runtime=nvidia'
            }
          }
          steps {
            sh './build.sh gcc-7'
          }
          post {
            always {
              step([
                $class: 'WarningsPublisher', canComputeNew: false, canResolveRelativePaths: false,
                defaultEncoding: '', excludePattern: '', healthy: '', includePattern: '', messagesPattern: '',
                parserConfigurations: [[parserName: 'GNU Make + GNU C Compiler (gcc)', pattern: 'build-gcc-7/make.out']],
                unHealthy: ''
              ])
            }
          }
        }
        stage('clang-5') {
          agent {
            docker {
              reuseNode true
              image 'braintwister/ubuntu-16.04-cuda-9.2-cmake-3.12-clang-5-conan-1.6'
              args '--runtime=nvidia'
            }
          }
          steps {
            sh './build.sh clang-5'
          }
          post {
            always {
              step([
                $class: 'WarningsPublisher', canComputeNew: false, canResolveRelativePaths: false,
                defaultEncoding: '', excludePattern: '', healthy: '', includePattern: '', messagesPattern: '',
                parserConfigurations: [[parserName: 'Clang (LLVM based)', pattern: 'build-clang-5/make.out']],
                unHealthy: ''
              ])
            }
          }
        }
      }
    }
    stage('Test') {
      parallel {
        stage('gcc-5') {
          agent {
            docker {
              reuseNode true
              image 'braintwister/ubuntu-16.04-cuda-9.2-cmake-3.12-gcc-5-conan-1.6'
              args '--runtime=nvidia'
            }
          }
          steps {
            sh 'cd build-gcc-5 && make test'
          }
          post {
            always {
              step([
                $class: 'XUnitBuilder',
                thresholds: [[$class: 'FailedThreshold', unstableThreshold: '1']],
                tools: [[$class: 'GoogleTestType', pattern: 'build-gcc-5/Testing/*.xml']]
              ])
            }
          }
        }
        stage('gcc-7') {
          agent {
            docker {
              reuseNode true
              image 'braintwister/ubuntu-16.04-cuda-9.2-cmake-3.12-gcc-7-conan-1.6'
              args '--runtime=nvidia'
            }
          }
          steps {
            sh 'cd build-gcc-7 && make test'
          }
          post {
            always {
              step([
                $class: 'XUnitBuilder',
                thresholds: [[$class: 'FailedThreshold', unstableThreshold: '1']],
                tools: [[$class: 'GoogleTestType', pattern: 'build-gcc-7/Testing/*.xml']]
              ])
            }
          }
        }
        stage('clang-5') {
          agent {
            docker {
              reuseNode true
              image 'braintwister/ubuntu-16.04-cuda-9.2-cmake-3.12-clang-5-conan-1.6'
              args '--runtime=nvidia'
            }
          }
          steps {
            sh 'cd build-clang-5 && make test'
          }
          post {
            always {
              step([
                $class: 'XUnitBuilder',
                thresholds: [[$class: 'FailedThreshold', unstableThreshold: '1']],
                tools: [[$class: 'GoogleTestType', pattern: 'build-clang-5/Testing/*.xml']]
              ])
            }
          }
        }
      }
    }
    stage('Doxygen') {
      agent {
        docker {
          reuseNode true
          image 'braintwister/ubuntu-16.04-cuda-9.2-cmake-3.12-gcc-7-conan-1.6-doxygen-1.8.13'
          args '--runtime=nvidia'
        }
      }
      steps {
        sh '''
            mkdir -p build-doc
            cd build-doc
            cmake ..
            make doc
        '''
        publishHTML( target: [
          allowMissing: false,
          alwaysLinkToLastBuild: false,
          keepAll: true,
          reportName: 'Doxygen',
          reportDir: 'build-doc/doxygen/html',
          reportFiles: 'index.html'
        ])
      }
    }
    stage('Deploy') {
      agent {
        docker {
          reuseNode true
          image 'braintwister/ubuntu-16.04-cuda-9.2-cmake-3.12-gcc-7-conan-1.6'
          args '--runtime=nvidia'
        }
      }
      steps {
        sh 'cd build-gcc-7 && make package'
      }
      post {
        success {
          archiveArtifacts artifacts: "build*/Pink*", fingerprint: true
        }
      }
    }
  }
  post {
    success {
      mail to: 'bernd.doser@h-its.org', subject: "SUCCESS: ${currentBuild.fullDisplayName}", body: "All fine."
    }
    failure {
      mail to: 'bernd.doser@h-its.org', subject: "FAILURE: ${currentBuild.fullDisplayName}", body: "Failed."
    }
  }
}
