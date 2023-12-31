/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/
#include <stdio.h>
#include <pangolin/pangolin.h>
#include "KeyFrameDisplay.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"
#include "util/FrameShell.h"

namespace dso
{
	namespace IOWrap
	{
		KeyFrameDisplay::KeyFrameDisplay()
		{
			originalInputSparse = 0;
			numSparseBufferSize = 0;
			numSparsePoints = 0;

			id = 0;
			active = true;
			camToWorld = SE3();

			needRefresh = true;

			my_scaledTH = 1e10;
			my_absTH = 1e10;
			my_displayMode = 1;
			my_minRelBS = 0;
			my_sparsifyFactor = 1;

			numGLBufferPoints = 0;
			bufferValid = false;
		}

		void KeyFrameDisplay::setFromF(FrameShell* frame, CalibHessian* HCalib)
		{
			id = frame->id;
			fx = HCalib->fxl();
			fy = HCalib->fyl();
			cx = HCalib->cxl();
			cy = HCalib->cyl();
			width = wG[0];
			height = hG[0];
			fxi = 1 / fx;
			fyi = 1 / fy;
			cxi = -cx / fx;
			cyi = -cy / fy;
			camToWorld = frame->camToWorld;
			needRefresh = true;
		}

		void KeyFrameDisplay::setFromKF(FrameHessian* fh, CalibHessian* HCalib)
		{
			// 设置当前关键帧的位姿以及内参数
			setFromF(fh->shell, HCalib);

			// add all traces, inlier and outlier points.
			int npoints = fh->immaturePoints.size() +
				fh->pointHessians.size() +
				fh->pointHessiansMarginalized.size() +
				fh->pointHessiansOut.size();

			if (numSparseBufferSize < npoints)
			{
				if (originalInputSparse != 0) delete originalInputSparse;
				numSparseBufferSize = npoints + 100;
				originalInputSparse = new InputPointSparse<MAX_RES_PER_POINT>[numSparseBufferSize];
			}

			numSparsePoints = 0;
			InputPointSparse<MAX_RES_PER_POINT>* pc = originalInputSparse;

			for (ImmaturePoint* p : fh->immaturePoints)
			{
				for (int i = 0; i < patternNum; i++)
					pc[numSparsePoints].color[i] = p->color[i];
				pc[numSparsePoints].u = p->u;
				pc[numSparsePoints].v = p->v;
				pc[numSparsePoints].idpeth = (p->idepth_max + p->idepth_min)*0.5f;
				pc[numSparsePoints].idepth_hessian = 1000;
				pc[numSparsePoints].relObsBaseline = 0;
				pc[numSparsePoints].numGoodRes = 1;
				pc[numSparsePoints].status = InputPointStatus::Immature;
				numSparsePoints++;
			}

			for (PointHessian* p : fh->pointHessians)
			{
				for (int i = 0; i < patternNum; i++)
					pc[numSparsePoints].color[i] = p->color[i];
				pc[numSparsePoints].u = p->u;
				pc[numSparsePoints].v = p->v;
				pc[numSparsePoints].idpeth = p->idepth_scaled;
				pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
				pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
				pc[numSparsePoints].numGoodRes = 0;
				pc[numSparsePoints].status = InputPointStatus::OK;
				numSparsePoints++;
			}

			for (PointHessian* p : fh->pointHessiansMarginalized)
			{
				for (int i = 0; i < patternNum; i++)
					pc[numSparsePoints].color[i] = p->color[i];
				pc[numSparsePoints].u = p->u;
				pc[numSparsePoints].v = p->v;
				pc[numSparsePoints].idpeth = p->idepth_scaled;
				pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
				pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
				pc[numSparsePoints].numGoodRes = 0;
				pc[numSparsePoints].status = InputPointStatus::Marginalized;
				numSparsePoints++;
			}

			for (PointHessian* p : fh->pointHessiansOut)
			{
				for (int i = 0; i < patternNum; i++)
					pc[numSparsePoints].color[i] = p->color[i];
				pc[numSparsePoints].u = p->u;
				pc[numSparsePoints].v = p->v;
				pc[numSparsePoints].idpeth = p->idepth_scaled;
				pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
				pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
				pc[numSparsePoints].numGoodRes = 0;
				pc[numSparsePoints].status = InputPointStatus::Out;
				numSparsePoints++;
			}
			assert(numSparsePoints <= npoints);

			camToWorld = fh->PRE_camToWorld;
			needRefresh = true;
		}

		KeyFrameDisplay::~KeyFrameDisplay()
		{
			if (originalInputSparse != 0)
				delete[] originalInputSparse;
		}

		// 刷新关键帧对应的点云
		// canRefresh:
		// scaledTH:
		// absTH:
		// mode:
		// minBS:
		// sparsity:
		bool KeyFrameDisplay::refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity)
		{
			if (canRefresh)
			{
				needRefresh = needRefresh ||
					my_scaledTH != scaledTH ||
					my_absTH != absTH ||
					my_displayMode != mode ||
					my_minRelBS != minBS ||
					my_sparsifyFactor != sparsity;
			}

			if (!needRefresh) return false;
			needRefresh = false;

			my_scaledTH = scaledTH;
			my_absTH = absTH;
			my_displayMode = mode;
			my_minRelBS = minBS;
			my_sparsifyFactor = sparsity;

			// 关键帧中并没有地图点此时不显示
			if (numSparsePoints == 0)
				return false;

			// make data
			Vec3f* tmpVertexBufferGlobal = new Vec3f[numSparsePoints*patternNum];
			Vec3b* tmpColorBufferGlobal = new Vec3b[numSparsePoints*patternNum];
			int vertexBufferNumPoints = 0;

			Vec3f landmarkLocal;
			Vec4f landmarkLocalHomo;
			Eigen::Matrix<float, 3, 4> m = camToWorld.matrix3x4().cast<float>();

			for (int i = 0; i < numSparsePoints; i++)
			{
				/* display modes:
				 * my_displayMode==0 - all pts, color-coded
				 * my_displayMode==1 - normal points
				 * my_displayMode==2 - active only
				 * my_displayMode==3 - nothing
				 */
				if (my_displayMode == 1 && originalInputSparse[i].status != 1 && originalInputSparse[i].status != 2) continue;
				if (my_displayMode == 2 && originalInputSparse[i].status != 1) continue;
				if (my_displayMode > 2) continue;

				if (originalInputSparse[i].idpeth < 0) continue;

				float depth = 1.0f / originalInputSparse[i].idpeth;
				float depth4 = depth * depth; depth4 *= depth4;
				float var = (1.0f / (originalInputSparse[i].idepth_hessian + 0.01));

				// TODO:数据乘以方差表示什么涵义
				if (var * depth4 > my_scaledTH)
					continue;

				// 当前点的方差较大表示点不可信
				if (var > my_absTH)
					continue;

				// 观测到该点的两帧间的基线如果比较短
				// 那么由这两帧得到的路标点的位置误差是比较大的
				if (originalInputSparse[i].relObsBaseline < my_minRelBS)
					continue;

				for (int pnt = 0; pnt < patternNum; pnt++)
				{
					if (my_sparsifyFactor > 1 && rand() % my_sparsifyFactor != 0) continue;
					int dx = patternP[pnt][0];
					int dy = patternP[pnt][1];

					landmarkLocal[0] = ((originalInputSparse[i].u + dx)*fxi + cxi) * depth;
					landmarkLocal[1] = ((originalInputSparse[i].v + dy)*fyi + cyi) * depth;
					landmarkLocal[2] = depth /** (1 + 2 * fxi * (rand() / (float)RAND_MAX - 0.5f))*/;

					landmarkLocalHomo = Vec4f::Ones();
					landmarkLocalHomo.block<3, 1>(0, 0) = landmarkLocal;
					tmpVertexBufferGlobal[vertexBufferNumPoints] = m * landmarkLocalHomo;
					
					if (my_displayMode == 0)
					{
						if (originalInputSparse[i].status == InputPointStatus::Immature)
						{
							tmpColorBufferGlobal[vertexBufferNumPoints][0] = 0;
							tmpColorBufferGlobal[vertexBufferNumPoints][1] = 255;
							tmpColorBufferGlobal[vertexBufferNumPoints][2] = 255;
						}
						else if (originalInputSparse[i].status == InputPointStatus::OK)
						{
							tmpColorBufferGlobal[vertexBufferNumPoints][0] = 0;
							tmpColorBufferGlobal[vertexBufferNumPoints][1] = 255;
							tmpColorBufferGlobal[vertexBufferNumPoints][2] = 0;
						}
						else if (originalInputSparse[i].status == InputPointStatus::Marginalized)
						{
							tmpColorBufferGlobal[vertexBufferNumPoints][0] = 0;
							tmpColorBufferGlobal[vertexBufferNumPoints][1] = 0;
							tmpColorBufferGlobal[vertexBufferNumPoints][2] = 255;
						}
						else if (originalInputSparse[i].status == InputPointStatus::Out)
						{
							tmpColorBufferGlobal[vertexBufferNumPoints][0] = 255;
							tmpColorBufferGlobal[vertexBufferNumPoints][1] = 0;
							tmpColorBufferGlobal[vertexBufferNumPoints][2] = 0;
						}
						else
						{
							tmpColorBufferGlobal[vertexBufferNumPoints][0] = 255;
							tmpColorBufferGlobal[vertexBufferNumPoints][1] = 255;
							tmpColorBufferGlobal[vertexBufferNumPoints][2] = 255;
						}
					}
					else
					{
						tmpColorBufferGlobal[vertexBufferNumPoints][0] = originalInputSparse[i].color[pnt];
						tmpColorBufferGlobal[vertexBufferNumPoints][1] = originalInputSparse[i].color[pnt];
						tmpColorBufferGlobal[vertexBufferNumPoints][2] = originalInputSparse[i].color[pnt];
					}
					vertexBufferNumPoints++;
					assert(vertexBufferNumPoints <= numSparsePoints * patternNum);
				}
			}

			if (vertexBufferNumPoints == 0)
			{
				delete[] tmpColorBufferGlobal;
				tmpColorBufferGlobal = NULL;
				delete[] tmpVertexBufferGlobal;
				tmpVertexBufferGlobal = NULL;

				return true;
			}

			numGLBufferGoodPoints = vertexBufferNumPoints;
			if (numGLBufferGoodPoints > numGLBufferPoints)
			{
				numGLBufferPoints = vertexBufferNumPoints * 1.3;
				vertexBufferGlobal.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW);
				colorBufferGlobal.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);
			}
			vertexBufferGlobal.Upload(tmpVertexBufferGlobal, sizeof(float) * 3 * numGLBufferGoodPoints, 0);
			colorBufferGlobal.Upload(tmpColorBufferGlobal, sizeof(unsigned char) * 3 * numGLBufferGoodPoints, 0);
			bufferValid = true;

			delete[] tmpColorBufferGlobal;
			tmpColorBufferGlobal = NULL;
			delete[] tmpVertexBufferGlobal;
			tmpVertexBufferGlobal = NULL;

			return true;
		}

		void KeyFrameDisplay::drawCam(float lineWidth, float* color, float sizeFactor)
		{
			if (width == 0)
				return;

			float sz = sizeFactor;

			glPushMatrix();

			Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
			glMultMatrixf((GLfloat*)m.data());

			if (color == 0)
			{
				glColor3f(1, 0, 0);	//BGR
			}
			else
				glColor3f(color[0], color[1], color[2]);

			glLineWidth(lineWidth);
			glBegin(GL_LINES);
			glVertex3f(0, 0, 0);
			glVertex3f(sz*(0 - cx) / fx, sz*(0 - cy) / fy, sz);
			glVertex3f(0, 0, 0);
			glVertex3f(sz*(0 - cx) / fx, sz*(height - 1 - cy) / fy, sz);
			glVertex3f(0, 0, 0);
			glVertex3f(sz*(width - 1 - cx) / fx, sz*(height - 1 - cy) / fy, sz);
			glVertex3f(0, 0, 0);
			glVertex3f(sz*(width - 1 - cx) / fx, sz*(0 - cy) / fy, sz);

			glVertex3f(sz*(width - 1 - cx) / fx, sz*(0 - cy) / fy, sz);
			glVertex3f(sz*(width - 1 - cx) / fx, sz*(height - 1 - cy) / fy, sz);

			glVertex3f(sz*(width - 1 - cx) / fx, sz*(height - 1 - cy) / fy, sz);
			glVertex3f(sz*(0 - cx) / fx, sz*(height - 1 - cy) / fy, sz);

			glVertex3f(sz*(0 - cx) / fx, sz*(height - 1 - cy) / fy, sz);
			glVertex3f(sz*(0 - cx) / fx, sz*(0 - cy) / fy, sz);

			glVertex3f(sz*(0 - cx) / fx, sz*(0 - cy) / fy, sz);
			glVertex3f(sz*(width - 1 - cx) / fx, sz*(0 - cy) / fy, sz);

			glEnd();
			glPopMatrix();
		}

		void KeyFrameDisplay::drawPC(float pointSize)
		{
			if (!bufferValid || numGLBufferGoodPoints == 0)
				return;

			glDisable(GL_LIGHTING);
			glPointSize(pointSize);

			colorBufferGlobal.Bind();
			glColorPointer(colorBufferGlobal.count_per_element, colorBufferGlobal.datatype, 0, 0);
			glEnableClientState(GL_COLOR_ARRAY);

			vertexBufferGlobal.Bind();
			glVertexPointer(vertexBufferGlobal.count_per_element, vertexBufferGlobal.datatype, 0, 0);
			glEnableClientState(GL_VERTEX_ARRAY);

			glDrawArrays(GL_POINTS, 0, numGLBufferGoodPoints);

			glDisableClientState(GL_VERTEX_ARRAY);
			vertexBufferGlobal.Unbind();
			glDisableClientState(GL_COLOR_ARRAY);
			colorBufferGlobal.Unbind();
		}

		void KeyFrameDisplay::savePC(std::ostream& out)
		{
			if (!bufferValid || numGLBufferGoodPoints == 0)
				return;
			
			float* vertexBuffer = new float[numGLBufferGoodPoints * 3];
			unsigned char* colorBuffer = new unsigned char[numGLBufferGoodPoints * 3];

			vertexBufferGlobal.Download(vertexBuffer, sizeof(float)*numGLBufferGoodPoints * 3, 0);
			colorBufferGlobal.Download(colorBuffer, sizeof(unsigned char)*numGLBufferGoodPoints * 3, 0);

			for (int it = 0; it < numGLBufferGoodPoints; ++it)
			{
				out << vertexBuffer[it * 3 + 0] << " "
					<< vertexBuffer[it * 3 + 1] << " "
					<< vertexBuffer[it * 3 + 2] << " "
					<< (int)colorBuffer[it * 3 + 0] << " "
					<< (int)colorBuffer[it * 3 + 1] << " "
					<< (int)colorBuffer[it * 3 + 2] << std::endl;
			}

			if (vertexBuffer)
			{
				delete[] vertexBuffer;
				vertexBuffer = NULL;
			}

			if (colorBuffer)
			{
				delete[] colorBuffer;
				colorBuffer = NULL;
			}
		}
	}
}